import re, ast, os, sys, glob, random, time, tempfile, math, subprocess, threading
try:
    import ujson
except ImportError:
    import json as ujson
import urllib.parse
import numpy as np
import io
import torch
from torch.utils.data import default_collate
import torchaudio
from typing import *
from dataclasses import dataclass, field
import transformers
from transformers.modeling_outputs import ModelOutput
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from functools import lru_cache
from io import BytesIO
from PIL import Image
import concurrent.futures as cf
from transformers.image_transforms import resize, center_crop, get_resize_output_image_size
from transformers.image_utils import PILImageResampling
from PIL import Image, ImageOps
from PIL import ImageFile
torch.set_num_threads(1)  # 限制torch的线程数 否则可能会卡住
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import base64
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
import cv2
try:
    import av
except ImportError:
    av = None
try:
    import imagesize
except ImportError:
    imagesize = None
from multiprocessing import Pool
try:
    from cairosvg import svg2png
except ImportError:
    svg2png = None
import hashlib

def _parse_bool_setting(raw_value, setting_name):
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{setting_name} must be one of 0/1, false/true, no/yes, off/on; "
        f"got {raw_value!r}"
    )

# RepAR-16x uses 16-pixel ViT patches. Qwen3-ViT still consumes patches in
# 2x2 block-major groups, so image dimensions must be aligned to 32 pixels.
# This alignment is not tokenizer downsampling: merge_size=1 keeps one output
# token per 16x16 input patch.
DEFAULT_PATCH_SIZE = 16
DEFAULT_SPATIAL_MERGE_SIZE = 2
DEFAULT_TOKENIZER_MERGE_SIZE = 1
DEFAULT_LLM_MERGER_SIZE = 1
IMAGE_FACTOR = (
    DEFAULT_PATCH_SIZE
    * DEFAULT_SPATIAL_MERGE_SIZE
    * DEFAULT_LLM_MERGER_SIZE
)
MIN_PIXELS = 9 * IMAGE_FACTOR * IMAGE_FACTOR       # 96 x 96
MAX_PIXELS = 3600 * IMAGE_FACTOR * IMAGE_FACTOR    # 1920 x 1920
MAX_RATIO = 200

# Keep the old T2I bucketresize contract, but derive bucket dimensions using
# the active tokenizer alignment instead of carrying over the Qwen2.5-ViT
# hard-coded 28-pixel table. The keys are the same 40 aspect-ratio buckets used
# by the old modeling.
ASPECT_RATIO_BUCKETS = (
    0.25, 0.26, 0.27, 0.28, 0.32, 0.33, 0.35, 0.40, 0.42, 0.48,
    0.50, 0.52, 0.57, 0.60, 0.68, 0.72, 0.78, 0.82, 0.88, 0.94,
    1.00, 1.07, 1.13, 1.21, 1.29, 1.38, 1.46, 1.67, 1.75, 2.00,
    2.09, 2.40, 2.50, 2.89, 3.00, 3.11, 3.62, 3.75, 3.88, 4.00,
)
SUPPORTED_BUCKET_RESOLUTIONS = (256, 512, 1024)

VIDEO_MIN_PIXELS = 128 * IMAGE_FACTOR * IMAGE_FACTOR
VIDEO_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
VIDEO_TOTAL_PIXELS = 24576 * IMAGE_FACTOR * IMAGE_FACTOR
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def split_text(text, match_regex):
    matches = list(re.finditer(match_regex, text))
    # 初始化结果列表
    result = []
    match_flag_list = []
    # 上一个匹配的结束位置
    last_end = 0
    # 遍历所有匹配项
    for match in matches:
        # 添加匹配项之前的部分
        if text[last_end:match.start()]:
            result.append(text[last_end:match.start()])
            match_flag_list.append(False)
        # 添加匹配项
        result.append(match.group(0))
        match_flag_list.append(True)
        # 更新上一个匹配的结束位置
        last_end = match.end()
    # 添加最后一个匹配项之后的部分
    if text[last_end:]:
        result.append(text[last_end:])
        match_flag_list.append(False)
    return result, match_flag_list


def read_video(image_path, max_frame_number, decode_way):
    if decode_way=='1fps':
        try:
            # print(image_path)
            vr = VideoReader(image_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), fps)]
            frames = vr.get_batch(frame_idx).asnumpy()
            cnt = len(frames)
            frame_times = range(cnt)
        except Exception as e:
            print(image_path)
            print('error is', e)
            return None
    elif decode_way=='key':
        try:
            with av.open(image_path) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'
                frames = []
                frame_times = []
                fps = int(stream.average_rate)
                cnt = 0
                for frame in container.decode(stream): # 关键帧存成image patch
                    image = np.array(frame.to_image())
                    frames.append(image)
                    frame_time = int(frame.time)
                    frame_times.append(frame_time)
                    cnt += 1
        except Exception as e:
            print('error is', e)
            return None
    if frames is None or len(frames)==0:
        return None
    if len(frames)>max_frame_number and max_frame_number>0:
        # 生成14个均匀间隔的索引
        indices = np.linspace(0, len(frames) - 1, max_frame_number, dtype=int)
        # 根据索引获取对应元素
        frames = frames[indices]
        frame_times = frame_times[indices]
    return frames, frame_times


class OmniImageProcessor:
    def __init__(self, config, **kwargs):
        self.config = config  # visual_config
        self.min_pixels = self.config.min_pixels if hasattr(self.config, 'min_pixels') else MIN_PIXELS
        self.max_pixels = self.config.max_pixels if hasattr(self.config, 'max_pixels') else MAX_PIXELS
        self.patch_size = self.config.patch_size if hasattr(self.config, 'patch_size') else DEFAULT_PATCH_SIZE
        self.temporal_patch_size = self.config.temporal_patch_size if hasattr(self.config, 'temporal_patch_size') else 2
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else DEFAULT_TOKENIZER_MERGE_SIZE
        self.spatial_merge_size = self.config.spatial_merge_size if hasattr(self.config, 'spatial_merge_size') else DEFAULT_SPATIAL_MERGE_SIZE
        # LLM 边界处的额外 spatial merger（在 visual_input_bridge 的 merge_size 之上再 merge_size 倍）.
        self.merger_merge_size = self.config.merger_merge_size if hasattr(self.config, 'merger_merge_size') else DEFAULT_LLM_MERGER_SIZE

    @property
    def image_factor(self):
        return self.patch_size * self.spatial_merge_size * self.merger_merge_size

    def _center_crop_resize(self, image, output_size):
        # output_size is int
        # Center crop the image to a square before resizing
        width, height = image.size
        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        image = image.crop((left, top, right, bottom))
        image = image.resize((output_size,output_size), PILImageResampling.BICUBIC)
        return image

    def _bucket_target_size(self, image, bucket_res):
        """Choose the old 40-bucket aspect ratio, aligned for this tokenizer."""
        if bucket_res not in SUPPORTED_BUCKET_RESOLUTIONS:
            raise ValueError(
                f"unsupported bucket_res={bucket_res}; "
                f"expected one of {SUPPORTED_BUCKET_RESOLUTIONS}"
            )
        width, height = image.size
        ratio = height / max(width, 1)
        target_ratio = min(ASPECT_RATIO_BUCKETS, key=lambda value: abs(value - ratio))
        factor = self.image_factor
        area = bucket_res * bucket_res
        target_h = max(factor, round_by_factor(math.sqrt(area * target_ratio), factor))
        target_w = max(factor, round_by_factor(math.sqrt(area / target_ratio), factor))
        return target_h, target_w

    def _bucket_resize(self, image, bucket_res):
        target_h, target_w = self._bucket_target_size(image, bucket_res)
        return image.resize((target_w, target_h), PILImageResampling.BICUBIC)

    _MAX_LOAD_PIXELS = 20_000_000
    def _safe_load_image(self, source):
        """Decode one image synchronously inside the outer bounded thread pool."""
        def _load():
            if isinstance(source, str):
                image = Image.open(source)
            else:
                try:
                    image = Image.open(BytesIO(source))
                except Exception:
                    if svg2png is None:
                        raise ImportError(
                            "cairosvg is required to decode SVG image inputs"
                        )
                    image = Image.open(BytesIO(svg2png(bytestring=source)))
            width, height = image.size
            if width * height > self._MAX_LOAD_PIXELS:
                scale = math.sqrt(self._MAX_LOAD_PIXELS / (width * height))
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    PILImageResampling.BICUBIC,
                )
            return image.convert("RGB")

        try:
            return _load()
        except Exception as exc:
            src = source if isinstance(source, str) else f"<bytes len={len(source)}>"
            print(
                f"[BAD_DATA] image DECODE FAILED ({type(exc).__name__}: {exc}): {src}",
                flush=True,
            )
            return None

    def image_transform(
        self,
        strseq,
        return_mm_data=True,
        fix_res: int = -1,
        bucket_res: int = -1,
        keep_original_image: bool = True,
    ):
        if not return_mm_data:
            raise ValueError("image_transform requires return_mm_data=True")
        image = self._safe_load_image(strseq)
        if image is None:
            raise ValueError("image decode failed or timed out")

        # 用于固定分辨率生成
        # TODO: 对于OCR数据怎么办?
        if fix_res > 0 and bucket_res > 0:
            raise ValueError("fixres and bucketresize are mutually exclusive")
        if fix_res>0:
            image = self._center_crop_resize(image, output_size=fix_res)
        elif bucket_res > 0:
            image = self._bucket_resize(image, bucket_res)
        original_image = image.copy() if keep_original_image else None
        image = np.array(image)
        image_org_size = image.shape[:2]  # 这里保存了图像的原始大小（高度和宽度），image.shape 返回图像的形状 (高度, 宽度, 通道数)，而 image.shape[:2] 提取了前两个值，即原始的高度和宽度。这个信息可以用于后续的对比或其他处理。

        # resize, crop, scale, normalize
        # 输出一个新的尺寸，这个尺寸通常是 (宽度, 高度) 格式，用于后续的图像调整操作，如缩放或裁剪。
        resized_height, resized_width = smart_resize(
            image_org_size[0], image_org_size[1],
            factor=self.image_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        output_size = (resized_height, resized_width)

        # 使用 resize 函数将图像调整到 output_size 大小。PILImageResampling.BICUBIC 指定使用双三次插值法来进行图像缩放，这种方法通常能够提供较好的图像质量。
        # image: 输入的图像数据，可以是 NumPy 数组或 PIL 图像对象；output_size: 目标大小，通常是一个二元组 (宽度, 高度)。这个尺寸可以是图像的绝对大小，也可以是相对于原始图像的比例；
        # resample: 可选的重采样方法，通常用于确定如何插值像素。例如，PILImageResampling.BICUBIC 表示使用双三次插值法，这是一种平滑的插值方法，常用于图像缩放。
        image = resize(image, output_size, PILImageResampling.BICUBIC)
        img = image.transpose(2, 0, 1)
        # 对图像进行归一化和标准化处理
        image = (
            img.astype(np.float32) / 255.0
            - np.array(self.config.image_mean, dtype=np.float32)[:, np.newaxis, np.newaxis]
        ) / np.array(self.config.image_std, dtype=np.float32)[:, np.newaxis, np.newaxis]
        # 处理成patch
        patches = image[np.newaxis, :]
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w), original_image


class OmniAudioProcessor:
    # 包含基本的音频特征抽取模块 + 输入数据解析模块
    def __init__(
        self,
        config,  # audio processor config
        **kwargs
    ):
        # make sure you have install 'conda install -c conda-forge 'ffmpeg<7'' for torchaudio
        assert(len(torchaudio.list_audio_backends()) > 0)
        self.config = config
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.config.n_fft // 2,
            num_mel_filters=self.config.num_mel_bins,
            min_frequency=0.0,
            max_frequency=self.config.sampling_rate / 2.0,
            sampling_rate=self.config.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.window = torch.hann_window(self.config.n_fft)

    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-6):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def zero_mean_unit_var_norm(x):
        return (x - x.mean()) / torch.sqrt(x.var() + 1e-8)

    def load_audio_waveform(self, uri, metadata=None, waveform_tensor=None, return_tensors=True, do_normalize=False):
        if metadata is None or waveform_tensor is None:
            metadata = torchaudio.info(uri)  # sample_rate, num_frames, num_channels, bits_per_sample, encoding=PCM_S
            waveform_tensor, _ = torchaudio.load(uri, normalize=True)
        assert(metadata.num_channels <= 2), "acoustic file with {} channels.".format(metadata.num_channels)  # whisper only accept mono channel audio

        if self.config.sampling_rate != metadata.sample_rate:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, metadata.sample_rate, self.config.sampling_rate)

        # downmix to mono channel https://trac.ffmpeg.org/wiki/AudioChannelManipulation
        if metadata.num_channels > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

        # normalized to zero mean (Qwen Audio没有处理 但Whisper官方实现)
        if do_normalize:
            waveform_tensor = self.zero_mean_unit_var_norm(waveform_tensor)

        if return_tensors:  # (channels, samples)
            return waveform_tensor
        else:
            return waveform_tensor.numpy()

    def split_with_overlap(self, waveform):  # 如果长度超过最大长度限制 分割为带overlap的多段
        channels, wave_samples = waveform.shape
        max_audio_samples = self.config.max_audio_seconds * self.config.sampling_rate
        if wave_samples <= max_audio_samples or self.config.split_overlap < 0:
            return [waveform]  # 没有超出最大长度or截断逻辑 统一返回list

        split_waveform, start = [], 0
        while start < wave_samples:  # 统一按秒数对齐overlap
            if start > int(self.config.sampling_rate * self.config.split_overlap):
                start -= int(self.config.sampling_rate * self.config.split_overlap)  # 0表示没有overlap，>0 overlap对应秒数
            end = min(start + max_audio_samples, wave_samples)
            if end - start>= self.config.n_fft: # 保证至少有一帧数据
                split_waveform.append(waveform[:, start:end])  # 注意这里可能会切割出特别短的片段 需要在预处理判断并丢弃
            start = end
        return split_waveform

    @classmethod
    def inference_output_length(cls, config, input_length):
        # for whisper + bridge
        kernel_size = config.kernel_size
        stride_size = config.stride_size
        avg_pooler = config.avg_pooler
        encoder_length = (input_length + 2 * (kernel_size // 2) - kernel_size) // 1 + 1  # conv layer1 with pad=1
        encoder_length = (encoder_length + 2 * (kernel_size // 2) - kernel_size) // stride_size + 1  # conv layer2 with pad=1
        if avg_pooler > 1:
            bridge_length = encoder_length // avg_pooler
        return encoder_length, bridge_length

    def extract_fbank_features(self, waveform):
        # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py
        channels, wave_samples = waveform.shape
        assert(wave_samples >= self.config.n_fft)
        valid_frame_nums = min(self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length, wave_samples // self.config.hop_length + 1)
        if wave_samples < self.config.max_audio_seconds * self.config.sampling_rate:
            waveform = torch.nn.functional.pad(waveform, (0, self.config.max_audio_seconds * self.config.sampling_rate - wave_samples), "constant", 0)
        else:
            waveform = waveform[:, :self.config.max_audio_seconds * self.config.sampling_rate]

        # window = torch.hann_window(self.config.n_fft)
        stft = torch.stft(waveform, self.config.n_fft, self.config.hop_length, window=self.window, return_complex=True)  # fft, len(wave) // n_fft // 2 + 1
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        log_spec = log_spec[0].numpy()  # (channel, filters, samples) -> (filters, samples)
        log_spec[:, valid_frame_nums:] = 0.0  # pad0

        return log_spec, valid_frame_nums

    def data_augment(self, feature: np.array, input_length, training=True):
        # reference https://arxiv.org/pdf/1904.08779
        def mask_start_indices(input_length, mask_length, min_masks, mask_prob):
            num_masked_span = int(mask_prob * input_length / mask_length + random.random())
            num_masked_span = max(num_masked_span, min_masks)
            start_indices = list(range(input_length - mask_length))
            random.shuffle(start_indices)
            start_indices = start_indices[:num_masked_span]
            return start_indices

        if not training or (self.config.mask_time_prob <= 0 and self.config.mask_feature_prob <= 0):
            return feature
        if input_length < self.config.mask_time_length * self.config.mask_time_min_masks + 1:
            return feature
        if self.config.num_mel_bins < self.config.mask_feature_length * self.config.mask_feature_min_masks + 1:
            return feature

        if self.config.mask_time_prob > 0:
            start_indices = mask_start_indices(input_length, self.config.mask_time_length, self.config.mask_time_min_masks, self.config.mask_time_prob)
            for start_idx in start_indices:
                feature[:, start_idx: start_idx + self.config.mask_time_length] = 0.0
        if self.config.mask_feature_prob > 0:
            start_indices = mask_start_indices(self.config.num_mel_bins, self.config.mask_feature_length, self.config.mask_feature_min_masks, self.config.mask_feature_prob)
            for start_idx in start_indices:
                feature[start_idx: start_idx + self.config.mask_feature_length, :] = 0.0

        return feature

@dataclass
class OmniProcessorOutput(ModelOutput):
    input_ids: Optional["List|torch.Tensor"] = None
    labels: Optional["List|torch.Tensor"] = None
    attention_mask: Optional["List|torch.Tensor"] = None
    position_ids: Optional["List|torch.Tensor"] = None
    seqlens: Optional["List|torch.Tensor"] = None  # 需要配合Omni Modeling使用
    # audio fields
    audios: Optional["List|torch.Tensor"] = None
    encoder_length: Optional["List|torch.Tensor"] = None
    bridge_length: Optional["List|torch.Tensor"] = None
    # image fields
    images: Optional["List|torch.Tensor"] = None
    patch_nums: Optional["List|torch.Tensor"] = None
    images_size: Optional["List|torch.Tensor"] = None
    crop_size: Optional["List|torch.Tensor"] = None
    images_grid: Optional["List|torch.Tensor"] = None
    # merge 后 LLM 视角的 token 网格大小 [(token_h, token_w), ...]
    token_size: Optional["List|torch.Tensor"] = None
    # 保存原始图像用于可视化
    original_images: Optional["List"] = None
    # 预编码 RVQ indices 通路：当走 cached://{shard}:{offset} 时, _get_image_cached
    # 直接读盘塞这里, 跳过 visual_tokenizer.encode. 模型 forward 收到 vision_tokens
    # 后走 get_visual_embed_given_tokens 分支(modeling_omni.py:725).
    vision_tokens: Optional["List|torch.Tensor"] = None
    # video fields
    videos: Optional["List|torch.Tensor"] = None
    videos_patch_nums: Optional["List|torch.Tensor"] = None
    videos_size: Optional["List|torch.Tensor"] = None
    videos_crop_size: Optional["List|torch.Tensor"] = None
    videos_grid: Optional["List|torch.Tensor"] = None
    # processor fields
    raw_text: Optional[str] = None
    index: Optional[int] = None
    # Keep this per-example instead of indexing a side list during packing.
    # Reader-side filtering and asynchronous completion can otherwise make the
    # packed result's index diverge from the temporary batch list.
    # ModelOutput requires every field after the first to default to None.
    has_eos_loss: Optional[bool] = None

    def concatenate(self, other):  # 仅限list使用
        def concat_one(a, b):
            if a is None and b is None:
                return None
            elif a is None and b is not None:
                return b
            elif a is not None and b is None:
                return a
            else:
                return a + b
        return OmniProcessorOutput(
            input_ids=concat_one(self.input_ids, other.input_ids),
            labels=concat_one(self.labels, other.labels),
            audios=concat_one(self.audios, other.audios),
            encoder_length=concat_one(self.encoder_length, other.encoder_length),
            bridge_length=concat_one(self.bridge_length, other.bridge_length),
            images=concat_one(self.images, other.images),
            images_grid=concat_one(self.images_grid, other.images_grid),
            patch_nums=concat_one(self.patch_nums, other.patch_nums),
            token_size=concat_one(self.token_size, other.token_size),
            original_images=concat_one(self.original_images, other.original_images),  # 添加原始图像拼接
            vision_tokens=concat_one(self.vision_tokens, other.vision_tokens),

            videos=concat_one(self.videos, other.videos),
            videos_grid=concat_one(self.videos_grid, other.videos_grid),
            videos_patch_nums=concat_one(self.videos_patch_nums, other.videos_patch_nums),

            position_ids=concat_one(self.position_ids, other.position_ids),
            seqlens=concat_one(self.seqlens, other.seqlens),
            images_size=concat_one(self.images_size, other.images_size),
            videos_size=concat_one(self.videos_size, other.videos_size),
            index=self.index,  # concat保持index不变
            has_eos_loss=self.has_eos_loss,
        )

class OmniMMProcessor(object):
    def __init__(self,
                tokenizer: transformers.PreTrainedTokenizer,
                config,
                training,
                relative_path=None,
                parallel=None,
                **kwargs,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.audio_processor = None
        if hasattr(config, 'audio_config') and getattr(config.audio_config, 'enable', True):
            self.audio_processor = OmniAudioProcessor(config.audio_config)
        self.visual_processor = None
        if hasattr(config, "visual_config"):
            self.visual_processor = OmniImageProcessor(config.visual_config)
        self.video_processor = None
        if hasattr(config, "video_config"):
            self.video_processor = OmniImageProcessor(config.video_config)
        self.training = training
        # Caption-T2I classifier-free guidance: at training time, drop the
        # non-trainable text condition with this probability.  The unconditional
        # branch is represented by an empty caption; class-condition tokens are
        # intentionally unsupported in this T2I-only modeling.
        self.image_cfg_drop_ratio = float(
            getattr(getattr(config, 'visual_config', config), 'image_cfg_drop_ratio', 0.0)
        )
        self.image_cfg_uncond_token_id = int(
            getattr(
                getattr(config, "visual_config", config),
                "image_cfg_uncond_token_id",
                0,
            )
        )
        self.relative_path = relative_path
        self.parallel = parallel
        honor_preprocess_env = os.environ.get(
            "OMNI_HONOR_JSON_PREPROCESS", ""
        ).strip()
        if honor_preprocess_env:
            self.honor_json_preprocess = _parse_bool_setting(
                honor_preprocess_env, "OMNI_HONOR_JSON_PREPROCESS"
            )
        else:
            self.honor_json_preprocess = bool(
                getattr(
                    getattr(config, "visual_config", config),
                    "honor_json_preprocess",
                    True,
                )
            )
        # audio tag
        self.audio_start_tag = None
        self.audio_end_tag = None
        self.audio_pad_tag = None
        self.audio_delim_tag = None
        self.audiotext_start_tag = None
        self.audiotext_end_tag = None
        self.audiotext_pad_tag = None
        self.audiogen_start_tag = None
        self.audiogen_end_tag = None
        if hasattr(self.config, "audio_config"):
            self.audio_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_start_token_id)
            self.audio_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_end_token_id)
            self.audio_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_pad_token_id)
            self.audio_delim_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_delim_token_id)
            self.audiotext_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiotext_start_token_id)
            self.audiotext_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiotext_end_token_id)
            self.audiotext_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiotext_pad_token_id)
            self.audiogen_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiogen_start_token_id)
            self.audiogen_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiogen_end_token_id)
        # image tag
        self.image_start_tag = None
        self.image_end_tag = None
        self.image_pad_tag = None
        self.video_start_tag = None
        self.video_end_tag = None
        # videoframe tag只是为了兼容图片帧作为输入的情况，没有token id，在抽取视频帧的时候，会将这个替换成image tag的start、end
        self.videoframe_start_tag = '<videoframe_start_omni>'
        self.videoframe_end_tag = '<videoframe_end_omni>'
        if hasattr(self.config, "visual_config"):
            # special token for start_tag
            self.image_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_start_token_id)
            # special token for end_tag
            self.image_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_end_token_id)
            # special token for pad_tag
            self.image_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_pad_token_id)
            self.image_line_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_line_token_id)
            self.image_delimiter_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_delimiter_token_id)
        if hasattr(self.config, "video_config"):
            self.video_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_start_token_id)
            self.video_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_end_token_id)
            self.image_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_start_token_id)
            self.image_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_end_token_id)
            self.image_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_pad_token_id)
            self.video_place_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_place_token_id)

            self.frame_pattern = getattr(self.config.video_config, 'frame_pattern', '<frame>')

        # any-resolution / fix-resolution 注入开关
        if hasattr(self.config, "visual_config"):
            self.enable_image_token_size_prefix = getattr(
                self.config.visual_config, "enable_image_token_size_prefix", False)
            if self.enable_image_token_size_prefix:
                self.image_token_size_prefix_start_tag = self.tokenizer.convert_ids_to_tokens(
                    self.config.visual_config.image_token_size_prefix_start_id)
                self.image_token_size_prefix_end_tag = self.tokenizer.convert_ids_to_tokens(
                    self.config.visual_config.image_token_size_prefix_end_id)
                self.trainable_image_token_size_prefix_when_trainable_image = getattr(
                    self.config.visual_config,
                    "trainable_image_token_size_prefix_when_trainable_image", False)
            self.enable_image_line_token = getattr(
                self.config.visual_config, "enable_image_line_token", False)
            # force_fix_res > 0 时无视数据中 <fixres_NNN> 标记，统一 center-crop+resize 到该边长
            # 优先级：env OMNI_FORCE_FIX_RES > visual_config.force_fix_res
            _env_ffr = os.environ.get("OMNI_FORCE_FIX_RES", "").strip()
            if _env_ffr:
                self.force_fix_res = int(_env_ffr)
            else:
                self.force_fix_res = int(getattr(self.config.visual_config, "force_fix_res", 0))
            self.force_bucket_res = int(os.environ.get(
                "OMNI_FORCE_BUCKET_RES",
                str(getattr(self.config.visual_config, "force_bucket_res", 0)),
            ) or 0)
            if self.force_bucket_res:
                if self.force_bucket_res not in SUPPORTED_BUCKET_RESOLUTIONS:
                    raise ValueError(f"unsupported forced bucket resolution: {self.force_bucket_res}")
                if self.force_fix_res > 0:
                    raise ValueError("forced square and bucket resolutions are mutually exclusive")
            self.config.visual_config.force_fix_res = self.force_fix_res
            self.config.visual_config.force_bucket_res = self.force_bucket_res
            # 预编码 RVQ indices 根目录, 当数据 jsonl 用 cached://<shard>:<offset> 协议时
            # 从该目录读取 .bin shard, 完全跳过 visual_tokenizer.encode.
            self.cached_rvq_root = os.environ.get("OMNI_CACHED_RVQ_ROOT", "").strip() or None
            # mmap shard 缓存(单进程内 LRU), 避免每次 sample 重新 open shard.
            self._cached_shard_mmap = {}

    def _get_audio(self, audio_info):
        try:
            audio_info = ujson.loads(audio_info)
            if 'path' in audio_info.keys():
                audio_uri, metadata, waveform_tensors = None, None, None
                audio_info['path'] = audio_info['path'].lstrip('/')
                audio_uri = os.path.join(self.relative_path, audio_info['path'])
                if not os.path.exists(audio_uri):
                    raise FileNotFoundError(audio_uri)

                waveforms = self.audio_processor.load_audio_waveform(audio_uri, metadata, waveform_tensors, True)
                waveforms = self.audio_processor.split_with_overlap(waveforms)  # 分割逻辑

                ret = OmniProcessorOutput()  # 默认初始化 audios字段为None
                for i, waveform in enumerate(waveforms):
                    audio, input_length = self.audio_processor.extract_fbank_features(waveform)
                    audio = self.audio_processor.data_augment(audio, input_length, self.training)
                    encoder_length, bridge_length = self.audio_processor.inference_output_length(self.config.audio_config, input_length)
                    if bridge_length <= 0:  # 过滤极端短数据 1. 如果len(waveforms)==1 ret=None; 2. len(waveforms)>1 则说明最后一段太短被抛弃
                        continue
                    current_ret = OmniProcessorOutput(
                        audios=[audio],
                        encoder_length=[encoder_length],
                        bridge_length=[bridge_length],
                        )
                    if ret.audios is None:
                        ret = current_ret
                    else:
                        ret = ret.concatenate(current_ret)  # 拼接多个切片
                return ret
            else:
                raise ValueError("can not find path in audio_info")
        except Exception as e:
            print("**** get audio error: {}......, info: {} *****".format(str(e)[:256], str(audio_info)))
        return OmniProcessorOutput()

    def _get_pure_audio_curriculum_loss(self, audio_info, audio_label, min_seconds=3.0, max_seconds=20.0):
        # mask开头至少x秒+随机y秒的音频，减轻 副语言信息 对LLM的负向
        # reference https://arxiv.org/pdf/2412.17048 Why Do Speech Language Models Fail to Generate Semantically Coherent Outputs? A Modality Evolving Perspective
        min_tokens = min_seconds * self.config.audio_config.sampling_rate // self.config.audio_config.hop_length + 1
        _, min_tokens = self.audio_processor.inference_output_length(self.config.audio_config, min_tokens)
        max_tokens = max_seconds * self.config.audio_config.sampling_rate // self.config.audio_config.hop_length
        _, max_tokens = self.audio_processor.inference_output_length(self.config.audio_config, max_tokens)
        min_tokens, max_tokens = int(min_tokens), int(max_tokens)
        try:
            audio_info = ujson.loads(audio_info)
            if not audio_info.get('curriculum', False):  # 数据不开启
                # print("NOT apply curriculum loss:\n{}".format(str(audio_info)))
                return audio_label
            if min_tokens >= len(audio_label):
                return audio_label  # 太短不开启（理论上不存在这类数据）
            masked_tokens = min(random.randint(min_tokens, max_tokens), len(audio_label) // 2)
            masked_tokens = max(min_tokens, masked_tokens)
            for mi in range(masked_tokens):
                if mi > 0:
                    audio_label[mi] = -100
            # print("DO apply curriculum loss:\n{}\nmin={},max={},masked={}\n{}".format(str(audio_info), min_tokens, max_tokens, masked_tokens, audio_label), flush=True)
        except Exception as e:
            print("*** _get_pure_audio_curriculum_loss error:={}".format(str(e)))
            pass
        return audio_label

    @staticmethod
    def _parse_preprocess(preprocess):
        """Return (fix_res, bucket_res) for the old preprocessing contract."""
        fix_res, bucket_res = -1, -1
        preprocess = (preprocess or "").strip()
        if preprocess.startswith("fixres_"):
            fix_res = int(preprocess.split("_", 1)[1])
        elif preprocess.startswith("bucketresize_"):
            bucket_res = int(preprocess.split("_", 1)[1])
        return fix_res, bucket_res

    @staticmethod
    def _strip_preprocess_tag(image_info):
        """Support both legacy prefix tags and the newer JSON field."""
        if not isinstance(image_info, str):
            return image_info, -1, -1
        match = re.match(r"^<((?:fixres|bucketresize)_\d+)>(.*)", image_info)
        if not match:
            return image_info, -1, -1
        tag, body = match.group(1), match.group(2)
        if body.startswith("{"):
            try:
                payload = ujson.loads(body)
                payload.setdefault("preprocess", tag)
                return ujson.dumps(payload), -1, -1
            except Exception:
                pass
        fix_res, bucket_res = OmniMMProcessor._parse_preprocess(tag)
        return body, fix_res, bucket_res

    def _get_image(self, image_info):
        # 预编码 RVQ indices 通路: cached://<shard_relpath>:<offset_imgs>.
        # 直接读盘塞 vision_tokens, 跳过整个 PIL/transform/encoder pipeline.
        if isinstance(image_info, str) and image_info.startswith("cached://"):
            if getattr(self, "force_bucket_res", 0) > 0:
                raise ValueError("forced bucket resize requires raw images, not cached visual tokens")
            return self._get_image_cached(image_info)

        image_info, fix_res_number, bucket_res_number = self._strip_preprocess_tag(image_info)
        # 全局 force_fix_res 模式：忽略数据里的 <fixres_NNN>，统一固定到该边长
        if getattr(self, "force_fix_res", 0) > 0:
            fix_res_number = self.force_fix_res
            bucket_res_number = -1

        try:
            is_local_path = isinstance(image_info, str) and (
                os.path.isabs(image_info) or not image_info.startswith("{")
            )
            if is_local_path and os.path.exists(image_info):
                image_data = image_info
            else:
                try:
                    image_info_dict = ujson.loads(image_info)
                except (ujson.JSONDecodeError, TypeError):
                    raise FileNotFoundError("local image not found: {}".format(image_info))
                else:
                    if "path" not in image_info_dict:
                        raise ValueError("can not find path in image_info")
                    if self.honor_json_preprocess:
                        dict_fix_res, dict_bucket_res = self._parse_preprocess(
                            image_info_dict.get("preprocess")
                        )
                        if dict_fix_res > 0 or dict_bucket_res > 0:
                            fix_res_number, bucket_res_number = (
                                dict_fix_res,
                                dict_bucket_res,
                            )
                    if getattr(self, "force_fix_res", 0) > 0:
                        fix_res_number = self.force_fix_res
                        bucket_res_number = -1

                    image_type = image_info_dict.get("type", "")
                    if image_type == "file":
                        image_data = image_info_dict["path"]
                    else:
                        image_info_dict["path"] = urllib.parse.unquote(
                            image_info_dict["path"].lstrip("/")
                        )
                        image_path = (
                            os.path.join(self.relative_path, image_info_dict["path"])
                            if self.relative_path
                            else image_info_dict["path"]
                        )
                        if not os.path.exists(image_path):
                            raise FileNotFoundError("local image not found: {}".format(image_path))
                        image_data = image_path

            # Apply after both legacy tags and JSON preprocess metadata so
            # a stage-wide bucket override cannot be reset by old fixres data.
            if getattr(self, "force_bucket_res", 0) > 0:
                fix_res_number, bucket_res_number = -1, self.force_bucket_res
            image_feat, org_size, image_list, original_image = (
                self.visual_processor.image_transform(
                    image_data,
                    fix_res=fix_res_number,
                    bucket_res=bucket_res_number,
                    keep_original_image=(
                        not self.training
                        or os.environ.get(
                            "OMNI_KEEP_ORIGINAL_IMAGES", ""
                        ).lower() in {"1", "true", "yes", "on"}
                    ),
                )
            )
            del image_data
            merge_length = self.visual_processor.merge_size**2
            merger_unit = self.visual_processor.merger_merge_size**2
            patch_nums = int(np.array(image_list).prod() // merge_length // merger_unit)
            # spatial_merge_size only determines Qwen-ViT input ordering. Token
            # downsampling is controlled by tokenizer merge_size (16x => 1,
            # 32x => 2) plus the optional LLM-side merger.
            grid_t, grid_h, grid_w = image_list
            token_downsample = (
                self.visual_processor.merge_size
                * self.visual_processor.merger_merge_size
            )
            token_h = grid_h // token_downsample
            token_w = grid_w // token_downsample
            if patch_nums != grid_t * token_h * token_w:
                raise ValueError(
                    "visual token layout mismatch: "
                    f"patch_nums={patch_nums}, grid_t={grid_t}, "
                    f"token_size=({token_h},{token_w})"
                )

            if org_size[0] * org_size[1] > 16**2:  # 极端小的图过滤
                return OmniProcessorOutput(
                        images=[image_feat],
                        patch_nums=[patch_nums],
                        crop_size=[image_list],
                        images_size= [org_size],
                        images_grid=[image_list],
                        token_size=[(token_h, token_w)],
                        original_images=(
                            [original_image] if original_image is not None else None
                        )
                        )
            else:
                print("**** image too small: {}, info: {} *****".format(str(org_size), str(image_info)))
                return OmniProcessorOutput()

        except Exception as e:
            print(
                "[BAD_DATA] get image error: {}......, info: {} *****".format(
                    str(e)[:256], str(image_info)
                ),
                flush=True,
            )
        return OmniProcessorOutput()

    def _cached_layout(self):
        """Per-image layout for the cached RVQ path, derived once from
        meta.json (written by encode_imagenet_rvq.py) + the visual_processor
        merge params. Mirrors the raw _get_image path so cached and live
        encoding agree.

        Returns dict: stride_tokens (rows on disk / image), depth,
        patch_nums (count of <img_pad> the LLM sees), token_h/token_w,
        grid_h/grid_w (ViT patch grid).

        Backward compatible: a 256-token / mms=1 / merge=1 tokenizer resolves
        to the previous hardcode (stride=256, patch_nums=256, grid=(16,16)).
        """
        cached = getattr(self, "_cached_layout_cache", None)
        if cached is not None:
            return cached

        # disk stride + depth: trust meta.json if present.
        stride_tokens, depth = 256, 8
        n_variants = 1
        try:
            meta_path = os.path.join(self.cached_rvq_root or "", "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = ujson.load(f)
                stride_tokens = int(meta.get("n_tokens", stride_tokens))
                depth = int(meta.get("depth", depth))
                n_variants = int(meta.get("n_variants", 1) or 1)
        except Exception as e:
            print(f"**** cached meta.json read failed ({e}); using defaults ****")

        vp = self.visual_processor
        merge_size = int(getattr(vp, "merge_size", 1))            # in-tokenizer bridge merge
        merger_merge_size = int(getattr(vp, "merger_merge_size", 1))  # LLM-side SpatialMerger
        patch_size = int(getattr(self.config.visual_config, "patch_size", 16))

        # ViT patch grid from fixed resolution (cached path is always fix-res).
        fix_res = int(getattr(self, "force_fix_res", 0)) or int(
            getattr(self.config.visual_config, "force_fix_res", 0))
        if fix_res > 0:
            grid_h = grid_w = fix_res // patch_size
        else:
            # fall back: infer square grid from disk stride + bridge merge.
            side = int(round(stride_tokens ** 0.5)) * merge_size
            grid_h = grid_w = side

        # LLM-visible token count + grid, same formula as the raw path.
        patch_nums = (grid_h * grid_w) // (merge_size ** 2) // (merger_merge_size ** 2)
        token_downsample = merge_size * merger_merge_size
        token_h = grid_h // token_downsample
        token_w = grid_w // token_downsample
        if patch_nums != token_h * token_w:
            raise ValueError(
                "cached visual token layout mismatch: "
                f"patch_nums={patch_nums}, token_size=({token_h},{token_w})"
            )

        cached = dict(
            stride_tokens=stride_tokens, depth=depth, patch_nums=patch_nums,
            token_h=token_h, token_w=token_w, grid_h=grid_h, grid_w=grid_w,
            n_variants=n_variants,
        )
        self._cached_layout_cache = cached
        print(f"[cached-layout] stride={stride_tokens} depth={depth} "
              f"patch_nums={patch_nums} token_size=({token_h},{token_w}) "
              f"grid=(1,{grid_h},{grid_w}) n_variants={n_variants}")
        return cached

    def _get_image_cached(self, image_info):
        """读取预编码的 RVQ indices, 跳过 visual_tokenizer.encode.

        URI 格式: cached://<shard_relpath>:<offset_in_imgs>
        shard 是 shape [N_imgs, stride_tokens, depth] int32 的 raw bin 文件,
        OMNI_CACHED_RVQ_ROOT 指向其上层目录.

        每张图的 token 布局 (stride/depth/patch_nums/grid) 由 _cached_layout()
        从 meta.json + visual_processor 的 merge 参数推出, 与 raw 路径一致;
        因此 64-token (8x8) / 256-token (16x16) 等不同 tokenizer 都能正确读取.
        """
        try:
            assert image_info.startswith("cached://")
            spec = image_info[len("cached://"):]
            shard_rel, _, offset_str = spec.rpartition(":")
            offset = int(offset_str)
            assert self.cached_rvq_root, (
                "cached:// URI seen but OMNI_CACHED_RVQ_ROOT is unset")
            shard_path = os.path.join(self.cached_rvq_root, shard_rel)
            mm = self._cached_shard_mmap.get(shard_path)
            if mm is None:
                # 用 numpy memmap, OS 自带 page cache; 多进程共享同 inode 的 mmap 页.
                mm = np.memmap(shard_path, dtype=np.int32, mode="r")
                self._cached_shard_mmap[shard_path] = mm
            lay = self._cached_layout()
            stride_tokens, depth = lay["stride_tokens"], lay["depth"]
            n_variants = int(lay.get("n_variants", 1) or 1)
            stride = stride_tokens * depth
            # Multi-variant (ten_crop) shards store n_variants blocks per image,
            # variant-major: [n_variants, stride_tokens, depth]. Pick a random
            # crop at train time (LlamaGen-style); deterministic variant 0 for
            # eval / inference so visualizations are stable.
            if n_variants > 1:
                variant = random.randrange(n_variants) if self.training else 0
                base = (offset * n_variants + variant) * stride
            else:
                base = offset * stride
            arr = np.asarray(mm[base : base + stride], dtype=np.int32).reshape(stride_tokens, depth).copy()
            # patch_nums / images_grid / token_size 需保持和 raw 路径一致, 让 input_ids
            # 中 <img_pad> 数量 = patch_nums (LLM 视角的 token 网格).
            return OmniProcessorOutput(
                vision_tokens=[arr],
                patch_nums=[lay["patch_nums"]],
                images_grid=[(1, lay["grid_h"], lay["grid_w"])],
                crop_size=[(1, lay["grid_h"], lay["grid_w"])],
                images_size=[(
                    lay["grid_h"] * int(self.visual_processor.patch_size),
                    lay["grid_w"] * int(self.visual_processor.patch_size),
                )],
                token_size=[(lay["token_h"], lay["token_w"])],
            )
        except Exception as e:
            print("**** cached image error: {}......, info: {} *****".format(
                str(e)[:256], str(image_info)))
            return OmniProcessorOutput()

    def _get_video_frame(self, video_frame_infos):
        try:
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, video_frame_infos)
            ret = OmniProcessorOutput()
            # 逐个解析
            for match in matches:
                video_frame_info = ujson.loads(match)
                # video_frame_info = ujson.loads(video_frame_info)
                if 'local' in video_frame_info.keys():
                    image_feat, org_size, image_list = self.video_processor.image_transform(video_frame_info['local'])
                elif 'path' in video_frame_info.keys() and os.path.exists(video_frame_info['path']):
                    image_feat, org_size, image_list = self.video_processor.image_transform(video_frame_info['path'])
                else:
                    raise ValueError("can not find any path in video_info")

                merge_length = self.video_processor.merge_size**2
                merger_unit = getattr(self.video_processor, 'merger_merge_size', 1)**2
                patch_nums = np.array(image_list).prod() // merge_length // merger_unit

                if org_size[0] * org_size[1] > 16**2:  # 极端小的图过滤
                    ret = ret.concatenate(
                            OmniProcessorOutput(
                                videos=[image_feat],
                                videos_patch_nums=[patch_nums],
                                videos_crop_size=[image_list],
                                videos_size= [org_size],
                                videos_grid=[image_list]
                            )
                        )
                else:
                    print("**** video too small: {}, info: {} *****".format(str(org_size), str(video_frame_info)))
            return ret

        except Exception as e:
            print("**** get video error: {}, info: {} *****".format(str(e)[:256], str(video_frame_info)))
        return OmniProcessorOutput()

    # 读取视频
    def _get_vision_obj_byte(self, source, path):
        vision_obj_byte = None
        if source == "local":
            if os.path.exists(path):
                vision_obj_byte = open(path, "rb").read()
            else:
                vision_obj_byte = None
        if source == "base64":
            vision_obj_byte = base64.b64decode(path)
        return vision_obj_byte

    # 将视频切分为帧，保存至子目录中
    def _split_video_to_frames(self, video_info, max_frame_number=-1, decode_way="1fps"):
        if decode_way=='1fps':
            frame_suffix = f'_frames'
        elif decode_way=='key':
            frame_suffix = f'_keyframes'
        else:
            raise ValueError('unvalid decode way!!!')

        server = "local"
        if 'local' in video_info.keys():
            # 本地路径
            video_path = video_info['local']
            # 帧保存本地路径
            frame_path = video_path[:video_path.rfind('.')] + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('local', video_path)
        elif 'base64' in video_info.keys():
            md5 = hashlib.md5(video_info['base64'].encode('utf-8')).hexdigest()
            if self.relative_path is not None:
                video_path = os.path.join(self.relative_path, md5)
            else:
                video_path = os.path.join(os.getcwd(), md5)
            frame_path = video_path + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('base64', video_info['base64'])
        elif 'url' in video_info.keys():
            md5 = hashlib.md5(video_info['url'].encode('utf-8')).hexdigest()
            if self.relative_path is not None:
                video_path = os.path.join(self.relative_path, md5)
            else:
                video_path = os.path.join(os.getcwd(), md5)
            frame_path = video_path + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('url', video_info['url'])
        else:
            raise ValueError('unvalid video server !!!')
            return ""

        if mm_obj_byte is None: # 未读取到视频文件
            return ""
        if not os.path.exists(frame_path) or len(os.listdir(frame_path))==0:
            # 保存帧
            os.makedirs(frame_path, exist_ok=True)
            frames, frame_times = read_video(io.BytesIO(mm_obj_byte), max_frame_number=-1, decode_way=decode_way) #读取全部帧
            for frame_idx, frame in enumerate(frames):
                output_filename = os.path.join(frame_path, f"{frame_times[frame_idx]}.jpg")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_filename, frame)
        frame_paths = os.listdir(frame_path)

        # 选取帧
        frame_times = [int(filename.split('/')[-1].replace('.jpg', '')) for filename in frame_paths if filename.endswith('.jpg')] # 文件名对应秒数
        frame_times.sort() #从小到大排序
        frame_number = len(frame_times)
        if frame_number > max_frame_number:
            indices = np.linspace(0, frame_number - 1, max_frame_number, dtype=int)
        else:
            indices = np.linspace(0, frame_number - 1, frame_number, dtype=int)
        # 拼接模式
        replace_str = ""
        for frame_idx, idx in enumerate(indices):
            frame_time = frame_times[idx]  # frame_time表示帧对应的时间 单位为s 同时也是存储的文件名
            frame_dict = {"local": os.path.join(frame_path, f'{frame_time}.jpg')}
            frame_str = self.frame_pattern.format(frame_idx) if '{}' in self.frame_pattern else self.frame_pattern  # {}对应的是第几张图片
            frame_str = frame_str.replace('<TIMEIDX>', str(frame_time))  # TIMEIDX对应的是第几秒
            frame_str = frame_str.replace('<TIMESTAMP>', time.strftime("%H:%M:%S", time.gmtime(frame_time)))  # TIMESTAMP对应的是时间戳
            frame_str = frame_str.replace('<frame>', f'{self.image_start_tag}{json.dumps(frame_dict)}{self.image_end_tag}')
            replace_str += frame_str

        return replace_str

    def sample_frame(self,frames_str,max_frame = 32):
        def uniform_sample(lst, num_samples):
            if num_samples > len(lst):
                return lst
            interval = len(lst) / num_samples
            samples = [lst[int(i * interval)] for i in range(num_samples)]
            return samples
        p = rf'({self.image_start_tag}.*?{self.image_end_tag})'
        frames_str_split = re.split(p,frames_str)
        frame_idxs = [idx for idx in range(len(frames_str_split)) if self.image_start_tag in frames_str_split[idx]]
        sample_frame_idxs = set(uniform_sample(frame_idxs, max_frame))
        return ''.join([item for idx,item in enumerate(frames_str_split) if idx in sample_frame_idxs or self.image_start_tag not in frames_str_split[idx]])

    def _get_video_frame_str(self, video_info):
        try:
            if self.videoframe_start_tag in video_info:#如果是以视频帧的形式表示一个视频，则替换成image tag
                frames_str = video_info
                frames_str = frames_str.replace(self.videoframe_start_tag,self.image_start_tag).replace(self.videoframe_end_tag,self.image_end_tag)
                return self.sample_frame(frames_str, max_frame = self.config.video_config.max_frame_num)
            video_info = ujson.loads(video_info)
            # 获取包含多帧图像路径的字符串，最大帧数量max_frame_number
            frames_str = self._split_video_to_frames(video_info, max_frame_number=self.config.video_config.max_frame_num, decode_way=self.config.video_config.decode_way)
            return frames_str
        except Exception as e:
            print("**** get video error: {}, info: {} *****".format(str(e), str(video_info)))
        return ""

    def _get_audiotext(self, audiotext_info):
        try:
            ret = self._get_audio(audiotext_info)  # 重复取结果 cached result
            audiotext_info = ujson.loads(audiotext_info)
            if ('audiotext' not in audiotext_info) or ('task_type' not in audiotext_info) or (not audiotext_info['audiotext'].strip()) or (audiotext_info['task_type'] not in ['asr', 'tts', 'response','sent_align','random']):
                raise ValueError("can not find audiotext or task_type or has wrong audiotext/task_type value in audiotext_info")
            return ret, audiotext_info['audiotext'], audiotext_info['task_type'], audiotext_info
        except Exception as e:
            print("**** get audiotext error: {}, info: {} *****".format(str(e), str(audiotext_info)))
        return OmniProcessorOutput(), '', '', {}

    def _replace_image(self, image_text):
        image_info = re.sub(re.compile(self.image_start_tag + "|" + self.image_end_tag), '', image_text)
        ret = self._get_image(image_info)  # 重复取结果 cached result
        if ret.patch_nums is None:
            return ''
        # 任意分辨率下显式注入 token_size 前缀和行尾换行 token
        token_h, token_w = ret.token_size[0]
        if getattr(self, "enable_image_token_size_prefix", False):
            # <img_token_size>{token_h} {token_w}</img_token_size>
            image_token_size_prefix_str = (
                f"{self.image_token_size_prefix_start_tag}{token_h} {token_w}"
                f"{self.image_token_size_prefix_end_tag}"
            )
        else:
            image_token_size_prefix_str = None

        if getattr(self, "enable_image_line_token", False):
            image_str = self.image_start_tag
            for i in range(token_h):
                image_str += self.image_pad_tag * token_w
                if i < token_h - 1:
                    image_str += self.image_line_tag
            image_str += self.image_end_tag
        else:
            image_str = self.image_start_tag + self.image_pad_tag * ret.patch_nums[0] + self.image_end_tag
        return ret, image_str, image_token_size_prefix_str


    def _replace_video_frame(self, video_frame_text):
        video_frame_info = re.sub(re.compile(self.image_start_tag + "|" + self.image_end_tag), '', video_frame_text)
        ret = self._get_video_frame(video_frame_info)  # 重复取结果 cached result
        if ret.videos_patch_nums is None:
            return ''
        video_frame_str = [self.image_start_tag + self.video_place_tag * ret.videos_patch_nums[i] + self.image_end_tag for i in range(len(ret.videos_patch_nums))]
        return ret, ''.join(video_frame_str)


    def split_multimodal_chunk(self, text_list, mm_label_list, trainable_list, mtype='audio'):
        # 抽取text中的json格式音频/图像信息，读取并转化为特征，同时估计encoder token数，填入对应数量的pad token
        if (self.audio_start_tag != None) and (mtype == 'audio'):
            match_regex = re.compile(self.audio_start_tag + '.*?' + self.audio_end_tag,re.S)
            drop_regex = re.compile(self.audio_start_tag + "|" + self.audio_end_tag,re.S)
        elif (self.image_start_tag != None) and (mtype == 'image'):
            match_regex = re.compile(self.image_start_tag + '.*?' + self.image_end_tag,re.S)
            drop_regex = re.compile(self.image_start_tag + "|" + self.image_end_tag,re.S)
        elif (self.audiotext_start_tag != None) and (mtype == 'audiotext'):
            match_regex = re.compile(self.audiotext_start_tag + '.*?' + self.audiotext_end_tag,re.S)
            drop_regex = re.compile(self.audiotext_start_tag + "|" + self.audiotext_end_tag,re.S)
        elif (self.audiogen_start_tag != None) and (mtype == 'audiogen'):
            match_regex = re.compile(self.audiogen_start_tag + '.*?' + self.audiogen_end_tag,re.S)
            drop_regex = re.compile(self.audiogen_start_tag + "|" + self.audiogen_end_tag,re.S)
        elif (self.video_start_tag != None) and (mtype == 'video'):
            match_regex = re.compile(self.video_start_tag + '.*?' + self.video_end_tag,re.S)
            drop_regex = re.compile(self.video_start_tag + "|" + self.video_end_tag,re.S)
        else:
            raise ValueError("mtype not supportted!")
        new_text_list = []
        new_mm_label_list = []
        new_trainable_flag_list = []
        for text,mm_label,trainable in zip(text_list,mm_label_list,trainable_list):
            for t,m in zip(*split_text(text, match_regex)):
                new_trainable_flag_list.append(trainable)
                if m:
                    new_text_list.append(re.sub(drop_regex, '', t))
                    new_mm_label_list.append(mtype)
                else:
                    new_text_list.append(t)
                    new_mm_label_list.append(mm_label)
        return new_text_list, new_mm_label_list, new_trainable_flag_list

    def process_multimodal_chunk(self, text, mm_label, trainable):
        ret = OmniProcessorOutput()
        if mm_label == 'audio':
            ret = self._get_audio(text)
            if ret.bridge_length is not None:
                ret.input_ids = self.tokenizer.encode(self.audio_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audio_end_tag,add_special_tokens=False)
                ret.labels = [a if trainable else -100 for a in ret.input_ids]
            else:
                raise ValueError(f"Get audio data Failed at Process audio chunk {text}")
        elif mm_label == 'audiogen':
            ret = self._get_audio(text)
            if ret.bridge_length is not None:
                ret.input_ids = self.tokenizer.encode(self.audiogen_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audiogen_end_tag,add_special_tokens=False)
                _labels = [a if trainable else -100 for a in ret.input_ids]
                ret.labels = self._get_pure_audio_curriculum_loss(text, _labels)  # 减弱纯音频的学习难度
            else:
                raise ValueError(f"Get audio data Failed at Process audio chunk {text}")
        elif mm_label == 'image':
            replaced = self._replace_image(text)
            if not replaced:
                raise ValueError("Get image data Failed at Process image chunk")
            ret, input_str, image_token_size_prefix_str = replaced
            if input_str:
                _input_ids_image = self.tokenizer.encode(input_str, add_special_tokens=False)
                _labels_image = [a if trainable else -100 for a in _input_ids_image]
                if image_token_size_prefix_str is not None:
                    _input_ids_prefix = self.tokenizer.encode(image_token_size_prefix_str, add_special_tokens=False)
                    prefix_trainable = trainable and getattr(
                        self, "trainable_image_token_size_prefix_when_trainable_image", False)
                    _labels_prefix = [a if prefix_trainable else -100 for a in _input_ids_prefix]
                    ret.input_ids = _input_ids_prefix + _input_ids_image
                    ret.labels = _labels_prefix + _labels_image
                else:
                    ret.input_ids = _input_ids_image
                    ret.labels = _labels_image
            else:
                raise ValueError("Get image data Failed at Process image chunk")
        elif mm_label == 'video':
            frame_str = self.video_start_tag+self._get_video_frame_str(text)+self.video_end_tag
            ret, input_str = self._replace_video_frame(frame_str)
            if input_str:
                ret.input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
                ret.labels = [a if trainable else -100 for a in ret.input_ids]
            else:
                raise ValueError("Get video data Failed at Process video chunk")
        elif mm_label == 'audiotext':  # 这类数据音频和文本都训练
            ret, audiotext, task_type, full_info = self._get_audiotext(text)
            if ret.bridge_length is not None:
                audiotext_input_ids = self.tokenizer.encode(audiotext, add_special_tokens=False)
                if len(audiotext_input_ids) > sum(ret.bridge_length)-4 or len(audiotext_input_ids) > self.tokenizer.model_max_length-1:  # 过滤长文本短音频或超长文本，这种属于不正常
                    raise ValueError(f"Audio text too long or too short {len(audiotext_input_ids)}, please check audio and text length！ 【{audiotext}】")

                # 音频token 根据类型确定音频是否可训练
                if task_type=='tts': # 文本在前面
                    audiotext_input_ids = self.tokenizer.encode(audiotext, add_special_tokens=False)
                    ret.input_ids = self.tokenizer.encode(self.audiogen_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audiogen_end_tag,add_special_tokens=False)
                    ret.labels = [a if trainable  else -100 for a in ret.input_ids]
                    ret.input_ids = audiotext_input_ids + ret.input_ids
                    ret.labels = [-100 for _ in audiotext_input_ids] + ret.labels
                elif task_type== 'asr': # 文本在后面, 文本要训练
                    audiotext_input_ids = self.tokenizer.encode(audiotext, add_special_tokens=False)
                    ret.input_ids = self.tokenizer.encode(self.audio_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audio_end_tag,add_special_tokens=False)
                    ret.labels = [-100 for _ in ret.input_ids] # asr音频先不训练 #[a if trainable else -100 for a in ret.input_ids]

                    ret.input_ids = ret.input_ids + audiotext_input_ids
                    ret.labels = ret.labels + audiotext_input_ids
                elif task_type=='sent_align':
                    segments = full_info['segments']
                    input_ids = []
                    labels = []
                    if len(segments) != len(ret.bridge_length):
                        raise ValueError(f"Audio text segments length not equal audio bridge length! 【{text}】")
                    if any([v<=0 for v in ret.bridge_length]):
                        raise ValueError(f"Some Segment bridge length is 0! 【{text}】")
                    for i, (seg, seg_bridge_length) in enumerate(zip(segments, ret.bridge_length)):
                        textids = self.tokenizer.encode(seg['text'])
                        input_ids.extend(textids)
                        labels.extend(textids if i!=0 else [-100 for _ in textids]) # 双工文本训练第一句不训练
                        audioids = self.tokenizer.encode(self.audiogen_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * seg_bridge_length + self.tokenizer.encode(self.audiogen_end_tag,add_special_tokens=False)
                        input_ids.extend(audioids)
                        labels.extend([a if trainable else -100 for a in audioids])
                    ret.input_ids = input_ids
                    ret.labels = labels
                else:
                    raise ValueError(f"Task type not supportted! must in ['asr', 'tts', 'random','sent_align] but get {task_type} current input {text}")
            else:
                raise ValueError("Get audio file Failed at Process audiotext chunk")
        elif mm_label == 'text':
            ret.input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ret.input_ids) > self.tokenizer.model_max_length-1:  # 过滤长文本
                raise ValueError(f"Text too long, please check text length！ 【{text[:5]+'...'*6+text[-5:]}】")
            ret.labels = [a if trainable else -100 for a in ret.input_ids]
        else:
            raise ValueError(f"mm_label not supportted! must in ['audio', 'image', 'audiotext', 'text'] but get {mm_label}")
        return ret

    def process_one(self, text, index=0, raw_only=False, has_eos_loss=True):
        ret = OmniProcessorOutput(index=index, has_eos_loss=has_eos_loss)
        # Per-example RNG keeps CFG-drop decisions stable across worker count,
        # thread scheduling, and repeated debug runs.
        item_seed = int(
            hashlib.md5(text[:200].encode("utf-8", errors="ignore")).hexdigest(),
            16,
        ) % (2**32)
        item_rng = random.Random(item_seed)
        all_text_list = []
        all_mm_label_list = []
        all_trainable_flag_list = []
        # 处理预训练中的trainable标记
        text_list, match_flag = split_text(text, re.compile("<trainable_start>.*?<trainable_end>",re.S))
        if len(text_list) == 1:
            text = re.sub(re.compile("<trainable_start>|<trainable_end>",re.S), '', text_list[0])
            all_text_list.append(text)
            all_mm_label_list.append('text')
            all_trainable_flag_list.append(True)
        else:
            for text, match in zip(text_list, match_flag):
                text = re.sub(re.compile("<trainable_start>|<trainable_end>",re.S), '', text)
                if text.strip() == '':
                    continue  # 把多余的空格干掉
                all_text_list.append(text)
                all_mm_label_list.append('text')
                all_trainable_flag_list.append(match)
        # 处理多模态信息
        for mtype in self.config.multimodal:  # 循环获取音频 图像结果
            all_text_list, all_mm_label_list, all_trainable_flag_list = self.split_multimodal_chunk(all_text_list, all_mm_label_list, all_trainable_flag_list, mtype)
        if len(all_text_list) == 0:
            print(f"Process {text} chunk error: No valid Text data!!!!!")
            return OmniProcessorOutput(index=index, has_eos_loss=has_eos_loss)

        for text, mm_label, trainable in zip(all_text_list, all_mm_label_list, all_trainable_flag_list):
            try:
                if not self.config.need_llm and mm_label == "text": # 没有llm的情况，就不需要再解码text了
                    continue
                # Caption-T2I CFG: skip the non-trainable prompt for a subset
                # of text-image pairs, producing the unconditional branch.
                if self.training and self.image_cfg_drop_ratio > 0.0 and \
                        len(all_mm_label_list) == 2 and \
                        all_mm_label_list == ["text", "image"] and \
                        mm_label == "text" and not trainable and \
                        item_rng.random() < self.image_cfg_drop_ratio:
                    if self.image_cfg_uncond_token_id > 0:
                        ret = ret.concatenate(
                            OmniProcessorOutput(
                                input_ids=[self.image_cfg_uncond_token_id],
                                labels=[-100],
                            )
                        )
                    continue
                mret = self.process_multimodal_chunk(text, mm_label, trainable)
                ret = ret.concatenate(mret)
            except ValueError as e:
                tt = text[:24].replace('\n','<LF>')
                print(f"Process {tt if mm_label == 'text' else text} {mm_label} chunk error: {str(e)}")
                return OmniProcessorOutput(index=index, has_eos_loss=has_eos_loss)

        if raw_only:
            ret.raw_text = self.tokenizer.decode(ret.input_ids, skip_special_tokens=False)
            return ret
        return ret

    @torch.no_grad()
    def __call__(self, example, parallel=128):  # 支持预训练数据string，sft数据message， 以及batch推理数据listofstring 3种形式
        if isinstance(example, Dict):
            pass
        elif isinstance(example, str):
            return self.process_one(example)
        elif isinstance(example, List):  # batch推理 异步多线程处理
            with cf.ThreadPoolExecutor(min(parallel, len(example))) as executor:
                future_list = [executor.submit(self.process_one, di, idx) for idx, di in enumerate(example)]
                batch_data = [key.result() for key in cf.as_completed(future_list)]
            valid_num = sum([1 if x.input_ids is not None else 0 for x in batch_data])
            assert(valid_num == len(batch_data))  # 推理数据严格要求数量对齐
            batch_data = sorted(batch_data, key=lambda x: x.index)  # 保证顺序不变

            ret = OmniProcessorOutput()
            for i in range(len(batch_data)):
                ret = ret.concatenate(batch_data[i])
            self.tokenizer.padding_side = "left"
            max_len = min(max([len(x.input_ids) for x in batch_data]),self.tokenizer.model_max_length)
            padding_result = self.tokenizer.pad({"input_ids": [r.input_ids for r in batch_data]}, return_tensors='pt')
            ret.input_ids = padding_result["input_ids"]
            ret.attention_mask = padding_result["attention_mask"]  # batch推理不pack 不需要seqlens
            ret.labels = torch.LongTensor([(max(0,max_len - len(x.labels))*[-100] + x.labels)[:max_len] for x in batch_data])

            if ret.audios is not None:
                max_audios_len = max([x.shape[-1] for x in ret.audios])
                ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios])

                ret.encoder_length = default_collate(ret.encoder_length)
                ret.bridge_length = default_collate(ret.bridge_length)

            if ret.images is not None:
                ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.images]
                ret.patch_nums = default_collate(ret.patch_nums)

            if ret.videos is not None:
                ret.videos = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.videos]
                ret.videos_patch_nums = default_collate(ret.videos_patch_nums)

            return ret

        else:
            raise ValueError("example format supported yet")


    @torch.no_grad()
    def pack_batch_pretrain(self, raw_batch, max_sequence_length=None, parallel=8):
        if self.parallel is not None:
            parallel = self.parallel
        if max_sequence_length is None:
            max_sequence_length = self.tokenizer.model_max_length
        # 将N条数据pack为M条 max_sequence_length长度的数据, 每条数据包含所属的多模态输入
        assert isinstance(raw_batch, List)
        start_ts = time.time()
        if parallel > 1:
            with cf.ThreadPoolExecutor(max_workers=parallel) as executor:
            # with cf.ProcessPoolExecutor(max_workers=parallel) as executor:
                future_list = []
                for idx, json_text in enumerate(raw_batch):
                    try:  # 读取json
                        json_obj = ujson.loads(json_text.strip())
                    except:
                        try:
                            json_obj = ast.literal_eval(json_text.strip())
                        except:
                            print("parse json obj faild: {}....".format(json_text[:300]))
                            continue
                    if isinstance(json_obj, list):
                        content = json_obj[1]
                    elif "input" in json_obj.keys():
                        content = (json_obj["title"] if "title" in json_obj.keys() else "") + json_obj["input"]
                    else:
                        content = (json_obj["title"] if "title" in json_obj.keys() else "") + json_obj["content"]
                    has_eos_loss = True
                    if not isinstance(json_obj, list):  # 获取是否有eos loss
                        has_eos_loss = json_obj.get('eosloss', True)
                    future_list.append(
                        executor.submit(
                            self.process_one,
                            content,
                            idx,
                            False,
                            has_eos_loss,
                        )
                    )
                # Preserve submission order so packing and deterministic CFG
                # dropout do not depend on I/O completion order.
                batch_data = [future.result() for future in future_list]
        else:  # debug only
            batch_data = [self.process_one(ujson.loads(json_text.strip())['content'], idx) for idx, json_text in enumerate(raw_batch)]

        if (time.time() - start_ts) / (len(batch_data) + 1e-3) > 1.0:
            print('[WARNING] processing each data cost more than 1.0s')

        # packing 文本部分的输入，不做任何截断
        current_length, packed_output, output = 0, OmniProcessorOutput(input_ids=[], labels=[], position_ids=[], seqlens=[]), []
        empty_data = OmniProcessorOutput(input_ids=[], labels=[], index=0)
        for idx, bd in enumerate(batch_data + [empty_data]):  # 加空数据方便appedn最后一个数据到output，防止遗漏
            if bd.input_ids is None and idx < len(batch_data):
                continue  # 数据没取到 并且不是最后一个
            if (len(bd.input_ids) <= 0 or len(bd.input_ids) + 1 > max_sequence_length) and idx < len(batch_data):
                continue  # 太长的直接不要 并且不是最后一个
            if current_length + len(bd.input_ids) + 1 > max_sequence_length or idx == len(batch_data):
                pad_nums = max_sequence_length - current_length  # right padding
                packed_output.input_ids += [self.tokenizer.pad_token_id] * pad_nums
                packed_output.labels += [-100] * pad_nums
                packed_output.attention_mask = [1] * current_length + [0] * pad_nums
                packed_output.position_ids += [0] * pad_nums
                packed_output.seqlens += [0] * (max_sequence_length - len(packed_output.seqlens))
                output.append(packed_output)
                packed_output = OmniProcessorOutput(input_ids=[], labels=[], position_ids=[], seqlens=[])  # reset empty
            packed_output = packed_output.concatenate(bd)
            packed_output.input_ids.append(self.tokenizer.eos_token_id)  # </s>需要单独加
            if len(packed_output.input_ids) > 0 and bd.has_eos_loss:  # 部分数据的文本结束不加eos label
                packed_output.labels.append(self.tokenizer.eos_token_id)
            else:
                packed_output.labels.append(-100)

            packed_output.position_ids.extend(list(range(len(bd.input_ids) + 1)))
            packed_output.seqlens.append(len(bd.input_ids) + 1)

            current_length = len(packed_output.input_ids)
        return output

    @torch.no_grad()
    def collect_batch_pretrain(self, batch_data):
        ret = OmniProcessorOutput()
        for i in range(len(batch_data)):
            ret = ret.concatenate(batch_data[i])
        ret.input_ids = default_collate([np.asarray(x.input_ids, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)
        ret.labels = default_collate([np.asarray(x.labels, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)
        ret.attention_mask = default_collate([np.asarray(x.attention_mask, dtype=np.float32) for x in batch_data]).cuda(non_blocking=True)
        ret.position_ids = default_collate([np.asarray(x.position_ids, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)
        ret.seqlens = default_collate([np.asarray(x.seqlens, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)

        ret.raw_text = None
        if ret.audios is not None:
            max_audios_len = max([x.shape[-1] for x in ret.audios])
            ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios]).cuda(non_blocking=True)
            ret.encoder_length = default_collate(np.asarray(ret.encoder_length, dtype=np.int32)).cuda(non_blocking=True)
            ret.bridge_length = default_collate(np.asarray(ret.bridge_length, dtype=np.int32)).cuda(non_blocking=True)
        if ret.images is not None:
            ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32)).cuda(non_blocking=True) for image in ret.images]
            ret.patch_nums = default_collate(np.asarray(ret.patch_nums, dtype=np.int32)).cuda(non_blocking=True)
        if ret.vision_tokens is not None:
            # 预编码 RVQ indices 路径: 每张图返回 [N_tokens, depth] int32 array, 这里
            # 沿 dim=0 cat 成 [N_total, depth] int64 cuda tensor, 与 modeling_omni.py
            # `get_visual_embed_given_tokens` 期望的形状对齐.
            ret.vision_tokens = torch.from_numpy(
                np.concatenate([np.asarray(t, dtype=np.int64) for t in ret.vision_tokens], axis=0)
            ).cuda(non_blocking=True)
            # cached 路径下 ret.images 为 None,patch_nums 不会被上面的分支转 cuda;
            # 这里补上, 否则 modeling_omni 里 sum(patch_nums) 依赖 cuda tensor.
            if not isinstance(ret.patch_nums, torch.Tensor):
                ret.patch_nums = default_collate(np.asarray(ret.patch_nums, dtype=np.int32)).cuda(non_blocking=True)
        if ret.videos is not None:
            ret.videos = [torch.from_numpy(np.asarray(video, dtype=np.float32)).cuda(non_blocking=True) for video in ret.videos]
            ret.videos_patch_nums = default_collate(np.asarray(ret.videos_patch_nums, dtype=np.int32)).cuda(non_blocking=True)

        return ret

    @torch.no_grad()
    def collect_batch_sft(self, batch_data):
        # list of dict to dataclass
        batch_data = [OmniProcessorOutput(**bd) for bd in batch_data]
        ret = OmniProcessorOutput()
        for i in range(len(batch_data)):
            ret = ret.concatenate(batch_data[i])
        ret.input_ids = default_collate([np.asarray(x.input_ids, dtype=np.int64) for x in batch_data])
        ret.labels = default_collate([np.asarray(x.labels, dtype=np.int64) for x in batch_data])
        ret.position_ids = default_collate([np.asarray(x.position_ids, dtype=np.int64) for x in batch_data])
        ret.seqlens = default_collate([np.asarray(x.seqlens, dtype=np.int64) for x in batch_data])

        ret.raw_text = None
        if ret.audios is not None:
            max_audios_len = max([x.shape[-1] for x in ret.audios])
            ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios])
            ret.encoder_length = default_collate(np.asarray(ret.encoder_length, dtype=np.int32))
            ret.bridge_length = default_collate(np.asarray(ret.bridge_length, dtype=np.int32))
        if ret.images is not None:
            # 转换 每个image 为torch tensor
            ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.images]
            ret.patch_nums = default_collate(np.asarray(ret.patch_nums, dtype=np.int32))
        if ret.videos is not None:
            ret.videos = [torch.from_numpy(np.asarray(video, dtype=np.float32))  for video in ret.videos]
            ret.videos_patch_nums = default_collate(np.asarray(ret.videos_patch_nums, dtype=np.int32))

        ret = ret.__dict__
        del ret['images_size']
        del ret['videos_size']
        del ret['crop_size']
        del ret['videos_crop_size']
        del ret['raw_text']
        del ret['index']
        del ret['attention_mask']
        del ret['original_images']
        ret.pop('token_size', None)
        return ret
