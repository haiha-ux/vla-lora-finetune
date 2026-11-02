"""Small hand-written LoRA for nn.Linear layers (no peft dependency).

W x + (alpha / r) * B (A x), with A: (r, in) random, B: (out, r) zeros, so the adapter starts as an exact no-op.
"""
import re

import torch
from torch import nn

# SmolVLA: the VLM's text-model attention projections (the action expert is named `lm_expert`, not matched).
DEFAULT_TARGET = r"vlm\.model\.text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    @property
    def weight(self):  # callers read `proj.weight.dtype`
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32, target: str = DEFAULT_TARGET) -> list[str]:
    """Freeze every parameter of `model`, then wrap matching Linear layers with LoRA. Returns wrapped names."""
    for p in model.parameters():
        p.requires_grad = False
    pat, names = re.compile(target), []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and pat.search(name):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, LoRALinear(mod, r, alpha))
            names.append(name)
    if not names:
        raise ValueError(f"no Linear layer matches {target!r}")
    return names


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]) -> None:
    own = {k for k in model.state_dict() if "lora_" in k}
    if own != set(sd):
        raise KeyError(f"adapter keys differ: missing {sorted(own - set(sd))[:3]}, unexpected {sorted(set(sd) - own)[:3]}")
    model.load_state_dict(sd, strict=False)


def param_counts(model: nn.Module) -> dict[str, int]:
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": train, "total": total, "base_total": total - train}


def zero_b_names(sd: dict[str, torch.Tensor]) -> list[str]:
    """Adapter B matrices that are exactly zero. Legitimate only for layers whose output never reaches the loss
    (SmolVLA's last VLM layer: its q_proj/o_proj outputs are unused by the action expert, so they get no gradient)."""
    return [k for k, v in sd.items() if "lora_B" in k and not v.abs().any()]
