from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


@dataclass
class VLMConfig:
    model_name_or_path: str
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    use_flash_attention_2: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"

    def resolved_target_modules(self) -> List[str]:
        return list(self.target_modules or DEFAULT_LORA_TARGET_MODULES)


def _resolve_torch_dtype(torch_module, dtype_name: str):
    if dtype_name in ("", "auto", None):
        return "auto"
    if not hasattr(torch_module, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch_module, dtype_name)


def load_processor(model_name_or_path: str, trust_remote_code: bool = True):
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("transformers is required to load a VLM processor") from exc

    return AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )


def _build_model_kwargs(torch_module, cfg: VLMConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "device_map": cfg.device_map,
    }
    dtype = _resolve_torch_dtype(torch_module, cfg.torch_dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if cfg.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    if cfg.load_in_4bit:
        kwargs["load_in_4bit"] = True
    if cfg.load_in_8bit:
        kwargs["load_in_8bit"] = True
    return kwargs


def load_vlm_model(cfg: VLMConfig):
    try:
        import torch
        from transformers import AutoModelForImageTextToText
    except Exception as exc:
        raise RuntimeError("transformers and torch are required to load a VLM model") from exc

    kwargs = _build_model_kwargs(torch, cfg)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name_or_path,
            **kwargs,
        )
    except Exception:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_name_or_path,
            **kwargs,
        )
    return model


def attach_lora(model, lora_cfg: Optional[LoRAConfig] = None):
    lora_cfg = lora_cfg or LoRAConfig()
    if not lora_cfg.enabled:
        return model
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
    except Exception as exc:
        raise RuntimeError("peft is required to attach LoRA adapters") from exc

    peft_cfg = PeftLoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        bias=str(lora_cfg.bias),
        target_modules=lora_cfg.resolved_target_modules(),
        task_type=str(lora_cfg.task_type),
    )
    model = get_peft_model(model, peft_cfg)
    return model


def load_vlm_and_processor(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    lora_target_modules: Optional[Iterable[str]] = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    use_flash_attention_2: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[Any, Any]:
    cfg = VLMConfig(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_flash_attention_2=use_flash_attention_2,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    processor = load_processor(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model = load_vlm_model(cfg)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if use_lora:
        model = attach_lora(
            model,
            LoRAConfig(
                enabled=True,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                bias=lora_bias,
                target_modules=list(lora_target_modules) if lora_target_modules else None,
            ),
        )
    return model, processor


def print_trainable_parameter_summary(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    summary = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }
    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    return summary
