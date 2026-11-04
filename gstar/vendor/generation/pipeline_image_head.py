import inspect
import sys
import copy
import os
import torch
import inspect
import warnings
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Callable, Tuple, Dict, Any
import torch.distributed as dist
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torchaudio 
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    ModelOutput,
    GenerationConfig,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput,
    GenerationMixin,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    is_deepspeed_zero3_enabled,
    # is_torchdynamo_compiling,
    is_hqq_available,
    # is_quanto_available,
    DynamicCache,
    EncoderDecoderCache,
    logging
)
try:
    from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING
except ImportError:
    NEED_SETUP_CACHE_CLASSES_MAPPING = {}

# Try to import QUANT_BACKEND_CLASSES_MAPPING and QuantizedCacheConfig
# These may not be available in all transformers versions
try:
    from transformers.generation.utils import QUANT_BACKEND_CLASSES_MAPPING, QuantizedCacheConfig
except ImportError:
    QUANT_BACKEND_CLASSES_MAPPING = None
    QuantizedCacheConfig = None
from .rope2d import (
    MRoPERotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
    get_rope_index_3d
)

from tqdm import tqdm
# from transformers.generation.stopping_criteria import validate_stopping_criteria
import torch

def expand_input_ids_with_image_pad(input_ids: torch.Tensor, model_kwargs: dict, pad_token_id: int):

    batch, seq_len = input_ids.shape
    H = model_kwargs['images_grid'][0][1]  # 高度
    W = model_kwargs['images_grid'][0][2]  # 宽度
    added_length = H * W

    # 创建填充部分并拼接
    pad_part = torch.full(
        (batch, added_length),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device
    )
    new_input_ids = torch.cat([input_ids, pad_part], dim=1)

    return new_input_ids  # [batch, seq_len + H*W]

@dataclass
class GenerationImageHeadOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
        visual_sequences (`torch.LongTensor` of shape `(batch_size, sequence_length,code_book_size)`):
            The generated visual sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    vision_sequences: Optional[torch.LongTensor] = None
    vision_embeddings: Optional[Tuple[torch.FloatTensor]] = None


logger = logging.get_logger(__name__)
class GenerationImageHead(GenerationMixin):

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        streamer_unit: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = 1.0,
        image_cfg_scale: Optional[float] = None,
        refine_gamma: Optional[float] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        # transformers >=4.49 removed _validate_model_class (it became a no-op log warning); fall back gracefully.
        if hasattr(self, "_validate_model_class"):
            self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        finish_until_max_new_tokens = kwargs.pop("finish_until_max_new_tokens", False)
        if hasattr(self.processor, 'enable_image_token_size_prefix') and self.processor.enable_image_token_size_prefix:
            visual = self.config.visual_config
            merger_total = int(visual.spatial_merge_size) * int(getattr(visual, "merger_merge_size", 1) or 1)
            token_rows = int(kwargs["images_grid"][0][1]) // merger_total
            kwargs["max_new_tokens"] += token_rows - 1
            enable_image_token_size_prefix = True
        else:
            enable_image_token_size_prefix = False

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        # transformers >=4.49 removed _validate_assistant; assistant_model is always None here.
        if hasattr(self, "_validate_assistant"):
            self._validate_assistant(assistant_model, tokenizer, None)

        # 用于输入image的一些参数、
        model_kwargs['images']      = kwargs.pop("images", None)
        model_kwargs['patch_nums']  = kwargs.pop("patch_nums", None) 
        model_kwargs['images_grid'] = kwargs.pop("images_grid", None)
        # 1.5 check for anyRes generation enabled

        model_kwargs['enable_image_token_size_prefix'] = enable_image_token_size_prefix
        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        # if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"
        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )
                
                if QUANT_BACKEND_CLASSES_MAPPING is None or QuantizedCacheConfig is None:
                    raise ImportError(
                        "Quantized cache is not available in this version of transformers. "
                        "Please upgrade transformers or use a different cache_implementation."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                '''
                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )
                '''

                model_kwargs[cache_name] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs[cache_name] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if (streamer is not None or streamer_unit is not None) and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        # transformers >=4.49 dropped **kwargs from _get_stopping_criteria signature; pass only the supported args.
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer
        )
        # 10. go into different generation modes
        # 只支持了sample模式
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
        
            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            return GenerationImageHead._sample(self,
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria= None, # 生成音频的时候不能有stopping_criteria默认在音频eos结束
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                max_new_tokens=kwargs["max_new_tokens"],
                finish_until_max_new_tokens=finish_until_max_new_tokens,
                cfg_scale=cfg_scale,
                image_cfg_scale=image_cfg_scale,
                refine_gamma=refine_gamma,
                **model_kwargs,
            )
        else:
            raise NotImplementedError

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        finish_until_max_new_tokens: bool,
        max_new_tokens: int,
        cfg_scale:float,
        image_cfg_scale: Optional[float] = None,
        refine_gamma: Optional[float] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        NUM_CODEBOOKS = len(self.config.visual_quantizer_config.codebook_sizes)
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        assert stopping_criteria is None
        do_sample = generation_config.do_sample
        batch_size, cur_len = input_ids.shape
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        IMAGE_PAD_TOKEN_ID  = self.config.visual_config.image_pad_token_id
        VISION_END_TOKEN_ID = self.config.visual_quantizer_config.codebook_sizes

        # ============================================================
        # MTP (multi-token-prediction) mode: visual_config.merger_merge_size > 1
        # ------------------------------------------------------------
        # In merger ckpts the LLM operates on a *compressed* token grid: each
        # LM hidden expands via image_head.spatial_unmerger into MMS² spatial
        # RVQ-token positions, and visual_bridge_model.spatial_merger folds
        # them back to one hidden when feeding the next LM step. So:
        #
        #   1 LM step (1 image_pad in input_ids) ↔ MMS² RVQ positions in
        #   vision_sequences (block-major within the block).
        #
        # All MTP-aware code paths fan out by `mms_unit = mms²`. When mms == 1
        # (legacy: 63159 merger=1 / 74379 nobridge / non-merger SFT), every
        # block below collapses to the original behavior — the variable just
        # multiplies / divides by 1.
        # ============================================================
        _mms = int(getattr(self.config.visual_config, "merger_merge_size", 1) or 1)
        _mms_unit = _mms * _mms

        # spatial_causal head: the mms_unit spatial positions are sampled in a
        # forced AR order (spatial-major), NOT level-major like default/block_ar.
        # Each X(s,l)'s teacher-forcing input at a group boundary uses R(s-1) =
        # the FULL depth reconstruction of the previous spatial position, so we
        # must finish all `depth` levels of position s-1 before starting s.
        _ihc = getattr(self.config.visual_config, "image_head_config", None)
        _spatial_causal = bool(getattr(_ihc, "image_head_spatial_causal", False)) if _ihc is not None else False
        # New grouped heads share one generate API.  AR and DSpark-attn perform
        # their complete group-causal loop (with a tiny KV cache) inside
        # image_head.sample(); MaskGiT performs its iterative unmask there.
        # Legacy configs have no image_head_type and continue through the old
        # boolean-dispatch paths below.
        _explicit_head_type = (
            str(getattr(_ihc, "image_head_type", "")).lower()
            if _ihc is not None else ""
        )
        _grouped_head = _explicit_head_type in ("ar", "maskgit", "dspark_attn")
        if image_cfg_scale is not None:
            if _explicit_head_type not in ("ar", "dspark_attn"):
                raise ValueError("Image CFG requires an explicit AR or DSpark head")
            if batch_size % 3:
                raise ValueError("Image CFG requires full/image/empty prompt branches")
        # mask / MaskGit head: all `depth` levels are decoded in ONE image_head.sample()
        # call (few-step iterative unmask, schedule e.g. [1,2,2,3]) instead of the
        # level-major AR _iter_plan loop below. sample() returns the full
        # (B*mms_unit, depth) next_token_ids, so everything downstream is unchanged.
        _mask_head = bool(getattr(_ihc, "image_head_mask", False)) if _ihc is not None else False

        # vision_sequences accumulates RVQ-token positions, NOT LM steps. With
        # MTP, each LM step pushes mms_unit positions in block-major order.
        vision_sequences = torch.empty(batch_size, 0, NUM_CODEBOOKS, device=input_ids.device, dtype=torch.long)
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # transformers >=4.49 changed signature from (input_ids, model_kwargs) to (seq_length, device, model_kwargs).
        try:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        except TypeError:
            model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
        # print(f"Generate max len {max_length} current len {cur_len}")
        # 老版本的transformers
        # while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length) and cur_len < max_length:
        # if fix_max_max_length
        if "images_grid" in model_kwargs and model_kwargs['images_grid'] is not None:
            images_grid = model_kwargs['images_grid']
        else:
            images_grid = None
        num_new_tokens = 0 
        pbar = tqdm()
        position_ids_3d = None
        # 新增：用于按行控制 image newline token 的状态
        enable_image_token_size_prefix = model_kwargs.get("enable_image_token_size_prefix", False)
        images_grid = model_kwargs.get("images_grid", None)
        image_line_token_id = getattr(self.config.visual_config, "image_line_token_id", None)

        token_w = None
        tokens_in_row = None  # [batch]，记录每个样本当前行已经生成了多少个 image code token

        if (
            enable_image_token_size_prefix
            and images_grid is not None
            and image_line_token_id is not None
        ):
            # token_h / token_w are the COMPRESSED LM-side grid (1 LM image_pad
            # per token). Training-side processor_omni.py:752 computes:
            #   merger_total = visual_config.spatial_merge_size * merger_merge_size
            #   token_h = grid_h // merger_total
            # Older code hardcoded `// 2`; that happens to match the two ckpts
            # we've seen (sm=2 mms=1 OR sm=1 mms=2 → both merger_total=2) but
            # is wrong for any future combo (e.g. sm=2 mms=2 → //4). Compute
            # it from config so MTP / non-MTP both stay correct.
            _vc = self.config.visual_config
            _merger_total = int(getattr(_vc, "spatial_merge_size", 1)) * int(
                getattr(_vc, "merger_merge_size", 1) or 1
            )
            token_h = images_grid[0][1] // _merger_total
            token_w = images_grid[0][2] // _merger_total
            tokens_in_row = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device,) and cur_len < max_length:
            # ===== 判断这一 step 是否是 newline step（全 batch 同步换行）=====
            is_newline_step = False
            if (
                tokens_in_row is not None
                and token_w is not None
                and not this_peer_finished
            ):
                active_mask = (unfinished_sequences != 0)
                if active_mask.any():
                    # 所有未结束样本当前行都已经生成了 token_w 个 code，则本 step 作为 newline
                    is_newline_step = (tokens_in_row[active_mask] == token_w).all()

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Keep zero-based positions for each unpadded prompt. In CFG the
            # unconditional row is left-padded; using cache-wide arange positions
            # shifts its RoPE angles and changes BF16 sampling numerics.
            mask = model_kwargs.get("attention_mask")
            if mask is not None:
                positions = mask.long().cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 1)
                model_inputs["position_ids"] = positions[:, -model_inputs["input_ids"].shape[1]:].contiguous()
            else:
                model_inputs.pop("position_ids", None)

            # 这里应该放modeling里prepare_inputs_for_generation中，这里先外部处理
            model_inputs['vision_tokens'] = model_kwargs.get('vision_tokens', None)
            # RAR needs the absolute raster step after KV-cache slicing removes
            # the original image-start and prior image-pad tokens.  Do not pass
            # this RAR-only kwarg to older/non-RAR checkpoints whose forward()
            # signature predates RAR support.
            if bool(getattr(
                    getattr(self.config, "visual_config", None),
                    "rar_random_order", False)):
                model_inputs['rar_generation_step'] = int(num_new_tokens)
            # 对于图像编辑来说,首次输入会有这些内容
            model_inputs['images'] = model_kwargs.pop('images', None)
            model_inputs['patch_nums'] = model_kwargs.pop('patch_nums', None)
            model_inputs['images_grid'] = model_kwargs.pop('images_grid', None)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            if position_ids_3d is not None:
                if num_new_tokens == 0:
                    model_inputs['position_ids_3d'] = position_ids_3d[:, :, :cur_len]
                else:
                    model_inputs['position_ids_3d'] = position_ids_3d[:, :, cur_len:cur_len+1]
            else:
                model_inputs.pop('position_ids_3d', None)

            # forward（code / newline step 都要，newline 只是不用 image_head 产新 code）
            sig = inspect.signature(self.forward)
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            has_pos_3d = 'position_ids_3d' in sig.parameters
            if not (has_kwargs or has_pos_3d):
                model_inputs.pop('position_ids_3d', None)
            outputs = self(**model_inputs, return_dict=True)

            # When the LLM forward sees only a non-image-pad token (e.g. an
            # <img_newline> consumed at step W in any-res mode), modeling's
            # visual_trainable_mask is empty and it leaves vision_emb_for_infer
            # as None. The next normal-step then crashes when image_head tries
            # to reshape that None. Recover by taking the last LLM hidden state
            # from outputs.hidden_states (last layer, last position).
            if (
                outputs.vision_emb_for_infer is None
                and getattr(outputs, "hidden_states", None) is not None
                and len(outputs.hidden_states) > 0
            ):
                last_layer = outputs.hidden_states[-1]
                if isinstance(last_layer, torch.Tensor) and last_layer.dim() >= 2:
                    outputs.vision_emb_for_infer = last_layer[:, -1, :]

            # ====================== newline step：只插换行 token，不生成 code ======================
            if is_newline_step:
                full_vision_token = torch.full(
                    (batch_size, 1),
                    IMAGE_PAD_TOKEN_ID,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                active_mask = (unfinished_sequences != 0)
                full_vision_token[active_mask, 0] = image_line_token_id

                # 把 newline token 接到 input_ids 里（给下一步当上下文）
                input_ids = torch.cat([input_ids, full_vision_token], dim=-1)

                # 更新 KV cache 等，但不改 vision_tokens / vision_sequences / num_new_tokens
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                model_kwargs['vision_tokens'] = None
                # 换行后，未结束样本的行内计数清零
                if tokens_in_row is not None:
                    tokens_in_row[active_mask] = 0
                cur_len += 1
                # 不更新 num_new_tokens，不改 unfinished_sequences
                del outputs
                # 不调用 image_head，不推送到 streamer
                continue

            # ====================== 正常 code step（你的原始逻辑） ======================
            # 这里 next_token_ids 只存"真正 code"，不掺 image_line_token_id
            #
            # MTP shape note: image_head.spatial_unmerger expands the input
            # `outputs.vision_emb_for_infer` from (B, H) to (B*mms_unit, H), so
            # `visual_tokens` MUST have shape (B*mms_unit, depth) to match.
            # When mms == 1, mms_unit == 1 so this is just (B, depth) as before.
            next_token_ids = torch.zeros(batch_size * _mms_unit, NUM_CODEBOOKS, dtype=torch.long, device=input_ids.device)
            if num_new_tokens < 5 or num_new_tokens in (50, 100, 200):
                _ve = outputs.vision_emb_for_infer
                if isinstance(_ve, torch.Tensor):
                    _v = _ve.detach().float()
                    print(f"[DBG step={int(num_new_tokens)}] vision_emb shape={tuple(_v.shape)} "
                          f"std={_v.std().item():.4f} first5={_v.flatten()[:5].tolist()}")
                _mi = {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__)
                       for k, v in model_inputs.items() if v is not None}
                print(f"[DBG step={int(num_new_tokens)}] model_inputs: {_mi}")
                _vt_in = model_inputs.get('vision_tokens', None)
                if isinstance(_vt_in, torch.Tensor):
                    print(f"[DBG step={int(num_new_tokens)}] vision_tokens IN (prev step's RVQ) "
                          f"shape={tuple(_vt_in.shape)} values[batch0]={_vt_in[0].tolist()}")
                _pi = model_inputs.get('position_ids', None)
                _ii = model_inputs.get('input_ids', None)
                if isinstance(_pi, torch.Tensor) and isinstance(_ii, torch.Tensor):
                    print(f"[DBG step={int(num_new_tokens)}] input_ids[batch0]={_ii[0].tolist()} "
                          f"position_ids[batch0]={_pi[0].tolist()}")
            # spatial_causal: spatial-major AR over the mms_unit positions. Each
            # (_s, i) step re-forwards image_head with the partially-filled
            # next_token_ids and writes back ONLY position _s's level i — the
            # causal mask makes the still-zero future positions irrelevant to the
            # current logit. default/block_ar keep the level-major plan (_s=None
            # writes all mms_unit positions of level i at once).
            if _grouped_head or _mask_head:
                # MaskGit: decode all `depth` levels of every (mms_unit) spatial
                # position in one shot via few-step iterative unmask. sample()
                # handles CFG (cond/uncond row split + mirror) and logits_processor
                # internally and returns the full (B*mms_unit, depth) next_token_ids,
                # so the AR loop below is skipped (empty plan).
                next_token_ids = self.image_head.sample(
                    outputs.vision_emb_for_infer,
                    self.model.visual_bridge_model.embedding_layers,
                    cfg_scale=cfg_scale, mms_unit=_mms_unit, batch_size=batch_size,
                    do_sample=do_sample, logits_processor=logits_processor,
                    vision_sequences=vision_sequences, refine_gamma=refine_gamma,
                    **({"image_cfg_scale": image_cfg_scale} if image_cfg_scale is not None else {}),
                )
                _iter_plan = []
            elif _spatial_causal and _mms_unit > 1:
                _iter_plan = [(s, l) for s in range(_mms_unit) for l in range(NUM_CODEBOOKS)]
            else:
                _iter_plan = [(None, l) for l in range(NUM_CODEBOOKS)]
            for (_s, i) in _iter_plan:
                logits = self.image_head(
                    outputs.vision_emb_for_infer,
                    next_token_ids,
                    self.model.visual_bridge_model.embedding_layers
                )

                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large
                if i == 0:
                    next_token_logits = logits[i][:, :].clone()
                else:
                    next_token_logits = logits[i][:, :-1].clone()

                # MTP shape note: logits row count is B*mms_unit (one row per
                # spatial position). CFG splits cond/uncond rows along that
                # axis, so the half-size also fans out by mms_unit.
                # vision_sequences logits-processor warmup uses the cond half
                # of the previously-sampled spatial positions; we slice the
                # same way (first half rows = cond).
                _half = (batch_size * _mms_unit) // 2 if cfg_scale != 1.0 else batch_size * _mms_unit
                if cfg_scale != 1.0:
                    cond_logits, uncond_logits = next_token_logits.split(_half, dim=0)
                    next_token_logits = cfg_scale * (cond_logits - uncond_logits) + uncond_logits
                # logits_processor expects the running history per row; vision_sequences
                # currently has shape (batch_size, T_so_far_positions, depth). When
                # mms_unit > 1, T_so_far is num_new_tokens * mms_unit, but we still
                # only have batch_size rows because cond/uncond live in dim 0 of
                # vision_sequences (not fanned out). For the CFG case we therefore
                # repeat-interleave the cond half along dim 0 to match logits rows.
                _vs_cond = vision_sequences[:batch_size // 2 if cfg_scale != 1.0 else batch_size, :, i]
                if _mms_unit > 1:
                    _vs_cond = _vs_cond.repeat_interleave(_mms_unit, dim=0)
                next_token_scores = logits_processor(_vs_cond, next_token_logits)

                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                if i == 0 and (_s is None or _s == 0) and num_new_tokens in (0, 1, 2, 3, 50, 100, 200):
                    _logits = next_token_logits.detach().float()
                    _top = _logits.topk(5, dim=-1)
                    print(f"[DBG head cb0] step={int(num_new_tokens)} batch={batch_size} mms_unit={_mms_unit} "
                          f"raw_logits min={_logits.min().item():.3f} max={_logits.max().item():.3f} "
                          f"std={_logits.std().item():.3f} | "
                          f"top5_idx={_top.indices[0].tolist()} top5_val={[f'{v:.2f}' for v in _top.values[0].tolist()]} "
                          f"sampled={next_tokens[0].item()}")

                if cfg_scale != 1.0:
                    # next_tokens is (B*mms_unit/2,) — the cond half. Mirror it
                    # to fill uncond half so next_token_ids has full B*mms_unit.
                    next_tokens = next_tokens.repeat((2,))

                if _s is None:
                    next_token_ids[:, i] = next_tokens
                else:
                    # spatial-major: only commit position _s's level i this step.
                    _row_is_s = (
                        torch.arange(batch_size * _mms_unit, device=next_token_ids.device) % _mms_unit
                    ) == _s
                    next_token_ids[_row_is_s, i] = next_tokens[_row_is_s]

                if return_dict_in_generate and i == 0 and (_s is None or _s == 0):  # 只记录第一个码表的分数
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)
                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

            if synced_gpus and this_peer_finished:
                del outputs
                continue  # don't waste resources running the code we don't need

            pbar.update(1)
            if finish_until_max_new_tokens:
                unfinished_vision_sequences = num_new_tokens < max_new_tokens
            else:
                # update generated ids, model inputs, and length for next step.
                # MTP: next_token_ids has B*mms_unit rows. A sample is
                # "finished" only when ALL its mms_unit spatial positions
                # produced EOS in cb[0] (otherwise it ends mid-block which
                # leaves the partial block invalid). Reshape to (B, mms_unit)
                # along cb[0] and ANY-EOS-in-block triggers finish.
                _eos_per_pos = (next_token_ids[:, 0] == VISION_END_TOKEN_ID[0]).view(batch_size, _mms_unit)
                unfinished_vision_sequences = ~(_eos_per_pos.any(dim=1))

            unfinished_sequences = unfinished_sequences & unfinished_vision_sequences
            this_peer_finished = unfinished_sequences.max() == 0

            # code step：这里就是普通的视觉占位 PAD，不是 newline.
            # MTP note: each LM step still adds exactly ONE image_pad to
            # input_ids — the LM views the compressed (post-merger) grid.
            # The mms² fan-out only lives inside next_token_ids / RVQ space.
            full_vision_pad_token = torch.full(
                (batch_size, 1),
                IMAGE_PAD_TOKEN_ID,
                dtype=torch.long,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, full_vision_pad_token], dim=-1)
            # vision_sequences accumulates RVQ positions in block-major order.
            # next_token_ids: (B*mms_unit, depth). Reshape to (B, mms_unit, depth)
            # so we append mms_unit positions per LM step under each batch row.
            # When mms_unit == 1 this is identical to the legacy
            # `next_token_ids[:, None, :]` append.
            _step_chunk = next_token_ids.view(batch_size, _mms_unit, NUM_CODEBOOKS)
            vision_sequences = torch.cat([vision_sequences, _step_chunk], dim=1)

            if streamer is not None:
                streamer.put(next_token_ids.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # 给下一步 self(**model_inputs) 用的 vision_tokens 仍然是"纯 code".
            # MTP shape: feed the raw (B*mms_unit, depth) tensor — visual_bridge_model
            # internally runs spatial_merger to fold mms_unit → 1 per batch row,
            # producing (B, hidden) which lines up with 1 image_pad per LM step.
            # When mms_unit == 1 this is identical to the legacy (B, depth).
            #
            # unfinished mask must broadcast across the mms_unit fan-out so that
            # finished samples zero out all mms_unit positions, not just one.
            _unfinished_mtp = unfinished_sequences.long().repeat_interleave(_mms_unit, dim=0).unsqueeze(-1)
            model_kwargs['vision_tokens'] = next_token_ids * _unfinished_mtp

            # 一次 code step：未结束样本的当前行 code 数 +1
            if tokens_in_row is not None and token_w is not None:
                active_mask = (unfinished_sequences != 0)
                tokens_in_row[active_mask] += 1

            cur_len += 1
            num_new_tokens += 1
            del outputs


        # 末尾都拼上eos来方便后续处理
        # vision_sequences = torch.cat([vision_sequences, torch.LongTensor(VISION_END_TOKEN_ID).to(input_ids).expand(batch_size,1,-1)], dim=1)
        # if "enable_image_token_size_prefix" in model_kwargs and model_kwargs['enable_image_token_size_prefix']:
        #     bsz, seq_len, num_codebooks = vision_sequences.size()
        #     token_h, token_w = images_grid[0][1] // 2, images_grid[0][2] // 2
        #     rows = []
        #     idx = 0
        #     for h in range(token_h):
        #         # 这里如果后续你想完全依赖 input_ids 里的 newline 来切行，可以再根据新逻辑调整；
        #         # 现在先保持原实现，方便你后面按需要改。
        #         end_idx = idx + token_w
        #         if end_idx > idx:
        #             rows.append(vision_sequences[:, idx:end_idx, :])
        #         idx = end_idx
        #         if h < token_h - 1:
        #             idx += 1

        #     vision_sequences = torch.cat(rows, dim=1)

        vision_embeddings = []
        for b_idx in range(batch_size):
            vision_embeddings.append(self.model.visual_bridge_model(vision_sequences[b_idx]).unsqueeze(0))
        vision_embeddings = torch.cat(vision_embeddings, dim=0)
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerationImageHeadOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    vision_sequences = vision_sequences,
                    vision_embeddings = vision_embeddings,
                )
            else:
                return GenerationImageHeadOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    vision_sequences = vision_sequences,
                    vision_embeddings = vision_embeddings,
                )
        else:
            return input_ids, vision_sequences, vision_embeddings
