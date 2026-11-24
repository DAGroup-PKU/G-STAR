"""Optimized implementation of the G-STAR refiners shipped with the Hugging Face weights.

Pro's SR2x ``refiner/`` and Flash's ``gstar-flash-1.7b-refiner/`` exports carry the same RepAR Ideogram-4 DiT, Flux2
VAE and attention code with their own ``config.json``. This package holds that implementation with the fast paths
added (single-sample DiT path, fused kernels, FlashInfer sm90 attention, cuBLASLt GEMMs). The fast paths are on by
default and fall back automatically; ``GSTAR_FAST=0`` or ``GSTAR_FAST_REFINER=0`` loads the folder's own code with
``trust_remote_code`` instead (see ``gstar/fast.py``).

The packaged classes are used only for a folder whose code is one of the exports this implementation was verified
against (SHA-256 of every ``*.py`` file: Pro's SR2x ``refiner/``, Flash's g8 refiner, the one-step SR2x refiner
``oneStep-SR2x_refiner/``) and whose ``config.json`` selects a decode sampler verified for that code (native for the
first two, one_step and dmad4 for the one-step export); any other folder, sampler or modified code runs the
folder's own code. The samplers are implemented as the one-step export's code does (``_decode_refiner_from_zq``).
"""
import hashlib
import json
import os

MODEL_TYPE = "repar_ideogram4_unified"
_AUTO_MODEL = "modeling_unified_tokenizer.GStarRefiner"

# SHA-256 of the *.py files of the verified exports: the files all three share, then the others per export
_COMMON_CODE = {
    "attention.py": "fea79f26aeb82cd0862d2b78bfd40ffe73a4af9c213d76d6401ad471b3dddd7c",
    "base_model.py": "9af6eff0c222be901a7775f91cae342a5f9197bd8feb52f662734039a823fb4b",
    "blocks.py": "82eeecf4b17fd94b251054044e16f06b5028c4b162560a10c0a95062029a673f",
    "configuration_bar.py": "f6be0a6e204577734d03a63c02f587ab92b6290c02cf48962696187cd3da862f",
    "flow_match.py": "4ac295551f0ca2329a575e30f64340a30296f9349a3bc792d15c1d8cca9ebe7d",
    "flux2_vae.py": "ca43f112c4af50fe6fc6eb0da2fdaca9ddb45b7beded7ad2e4fbf7dc3d79fdf8",
    "gradient_checkpoint.py": "00f33981e4ac155d1ae757de2765703cd4881775b9cf8360627c709d631c2b21",
    "ideogram4_vae.py": "fba0eca997c8e8b0be28543d7946a1ff36daac4a0a5e0714a0cc7ccd07d55753",
    "modeling_bar.py": "a12248c396454cf48d505f666067b6df9b0b08c12a802c2fc47deeb1632828d2",
    "repar_common.py": "cda1c7e27a18cc439cb7353b0f50aee5e2196e2283bf4aff9eaf000d7b2439c1",
    "res_mod.py": "aecb12a6df8f13154e1c14a671565ec75bb052eec73e29aa6f73fd2d59104e31",
    "tokenizer.py": "45010e81687afde48839333abd2fd3a6ec838b98702a544c8b5c6d3b070f2494",
    "wavelet_color_fix.py": "f805d4ef0624cdc310739f840313412e926f9b922c08c11cf899b7afbbbeeabc",
}
# files the one-step export shares with the Pro export but not with the Flash one
_PRO_ONLY = {
    "ideogram4_dit.py": "508ec75107a7b98f366be7bca370cb65334004967e9b813e3fafa4e415a3e848",
    "repar_ideogram4_dit.py": "41b8c05e6230f21367655a638e3f8b587b3b93632034385be28dacd64ed2276a",
}
_VERIFIED_CODE = {
    "G-STAR Pro SR2x refiner": {**_COMMON_CODE,
        "configuration_unified_tokenizer.py": "f99013f74ff510297bbfff8b191015a1eec89ae3c0d119a9b9bc350b494d3d8d",
        "ideogram4_dit.py": "508ec75107a7b98f366be7bca370cb65334004967e9b813e3fafa4e415a3e848",
        "modeling_unified_tokenizer.py": "5f0bd0aecd6bf7caeae676deb195abf69226fd1cfff545a70e20f49e4038e21a",
        "quantizer.py": "7302d4319a56849a0c49ac5547972cf1a563e60ec6b33b6ba456534cba43ab0f",
        "repar_ideogram4_dit.py": "41b8c05e6230f21367655a638e3f8b587b3b93632034385be28dacd64ed2276a",
    },
    "G-STAR-Flash g8 refiner": {**_COMMON_CODE,
        "configuration_unified_tokenizer.py": "e3ad362792411beb4d1d5218c7873dd0db6049fd658156b5107d79818e6cc259",
        "ideogram4_dit.py": "47bae355738313791a9ae65f7b5cdf365f5ca4b92d7428a2c899f28387048012",
        "modeling_unified_tokenizer.py": "6efe5db5e71f11124affac4d43126b7d7e2b0f0c329475b1137bfc3735fb4a82",
        "quantizer.py": "7684da1753731bd264f83b7be18250373f312c11feec091bd7de6125739a8918",
        "repar_ideogram4_dit.py": "89cf9563e84c0402906519012d026d9e2e8fbee8e0f3d61711c1546bf3e71c52",
    },
    "G-STAR one-step SR2x refiner": {**_COMMON_CODE, **_PRO_ONLY,
        "configuration_unified_tokenizer.py": "331ad539980ea9126fbb20315dd1ace39e21cc3bb63abe72a935bb2cb991ce75",
        "modeling_unified_tokenizer.py": "fb894fe48993aed55bcb2048bcafe123b32f661e8f6b60406dea0c7ea078856d",
        "quantizer.py": "7684da1753731bd264f83b7be18250373f312c11feec091bd7de6125739a8918",
    },
}


# decode samplers each export's code was verified with (config.json "decode_sampler", absent = "native")
_VERIFIED_SAMPLERS = {
    "G-STAR Pro SR2x refiner": ("native",),
    "G-STAR-Flash g8 refiner": ("native",),
    "G-STAR one-step SR2x refiner": ("one_step", "dmad4"),
}


def _code_digest(tok_path):
    out = {}
    for name in sorted(os.listdir(tok_path)):
        p = os.path.join(tok_path, name)
        if name.endswith(".py") and os.path.isfile(p):
            with open(p, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def check(tok_path):
    """(True, export name) if tok_path holds a refiner export this implementation was verified against, else
    (False, reason)."""
    try:
        with open(os.path.join(tok_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        return False, f"config.json unreadable ({type(exc).__name__})"
    auto = cfg.get("auto_map") or {}
    if cfg.get("model_type") != MODEL_TYPE or not str(auto.get("AutoModel", "")).endswith(_AUTO_MODEL):
        return False, "not a G-STAR refiner export"
    sampler = cfg.get("decode_sampler", "native")
    try:
        digest = _code_digest(tok_path)
    except OSError as exc:
        return False, f"code unreadable ({type(exc).__name__})"
    for name, known in _VERIFIED_CODE.items():
        if digest == known:
            if sampler in _VERIFIED_SAMPLERS[name]:
                return True, name
            return False, f"decode_sampler {sampler!r} is not verified for the {name} code"
    return False, "its code is not one of the verified refiner exports"


def supports(tok_path):
    """tok_path holds a refiner export this implementation was verified against (see check)."""
    return check(tok_path)[0]


def load_refiner(tok_path, dtype):
    """GStarRefiner from tok_path's config.json and weights, built from this package's classes.

    Raises if the folder's weights do not map one-to-one onto these classes (the caller then loads the folder's
    own code)."""
    from .configuration_unified_tokenizer import GStarRefinerConfig
    from .modeling_unified_tokenizer import GStarRefiner
    config = GStarRefinerConfig.from_pretrained(tok_path)
    model, info = GStarRefiner.from_pretrained(tok_path, config=config, torch_dtype=dtype, output_loading_info=True)
    bad = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if bad:
        raise RuntimeError(f"weights do not match the packaged refiner implementation: "
                           f"{ {k: len(v) for k, v in bad.items()} }")
    return model
