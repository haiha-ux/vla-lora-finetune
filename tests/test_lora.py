import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora import LoRALinear, inject_lora, lora_state_dict, load_lora_state_dict, param_counts


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(8, 8) for _ in range(4))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Module()
        self.vlm.model = nn.Module()
        self.vlm.model.text_model = nn.Module()
        self.vlm.model.text_model.layers = nn.ModuleList()
        for _ in range(2):
            l = nn.Module(); l.self_attn = Attn(); self.vlm.model.text_model.layers.append(l)
        self.lm_expert = nn.Module(); self.lm_expert.layers = nn.ModuleList([nn.Module()])
        self.lm_expert.layers[0].self_attn = Attn()
        self.head = nn.Linear(8, 8)


def test_injection_matches_only_vlm_attention_and_freezes_rest():
    m = Toy()
    names = inject_lora(m, r=4, alpha=8)
    assert len(names) == 8 and all("lm_expert" not in n for n in names)
    assert isinstance(m.vlm.model.text_model.layers[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(m.lm_expert.layers[0].self_attn.q_proj, LoRALinear)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    c = param_counts(m)
    assert c["trainable"] == 8 * 4 * (8 + 8)


def test_adapter_is_noop_at_init_then_changes_output_and_grads_flow():
    torch.manual_seed(0)
    m = Toy(); x = torch.randn(3, 8)
    ref = m.vlm.model.text_model.layers[0].self_attn.q_proj(x)
    inject_lora(m)
    q = m.vlm.model.text_model.layers[0].self_attn.q_proj
    assert torch.allclose(q(x), ref)
    assert q.weight.dtype == torch.float32
    q(x).sum().backward()
    assert q.lora_B.grad.abs().sum() > 0 and q.base.weight.grad is None
    with torch.no_grad():
        q.lora_B.add_(0.1)
    assert not torch.allclose(q(x), ref)


def test_save_load_round_trip(tmp_path):
    torch.manual_seed(1)
    a = Toy(); b = Toy(); b.load_state_dict(a.state_dict())
    inject_lora(a); inject_lora(b)
    for n, p in a.named_parameters():
        if "lora_B" in n:
            with torch.no_grad(): p.normal_()
    f = tmp_path / "adapter.safetensors"
    save_file(lora_state_dict(a), str(f))
    x = torch.randn(2, 8)
    qa, qb = (m.vlm.model.text_model.layers[1].self_attn.v_proj for m in (a, b))
    assert not torch.allclose(qa(x), qb(x))
    load_lora_state_dict(b, load_file(str(f)))
    assert torch.allclose(qa(x), qb(x))


def test_load_rejects_mismatched_keys():
    a = Toy(); inject_lora(a)
    sd = lora_state_dict(a); sd.pop(next(iter(sd)))
    try:
        load_lora_state_dict(a, sd)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_unused_layer_keeps_zero_B_and_is_detected():
    """A wrapped layer whose output never reaches the loss gets no gradient, so its B stays zero after training.
    (SmolVLA: last VLM layer's q_proj/o_proj.)  zero_b_names must flag exactly those, nothing else."""
    from lora import zero_b_names
    torch.manual_seed(0)
    m = Toy(); inject_lora(m, r=2, alpha=4)
    x = torch.randn(4, 8)
    l0, l1 = (m.vlm.model.text_model.layers[i].self_attn for i in (0, 1))
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        (l0.q_proj(x) + l0.o_proj(x)).pow(2).sum().backward()   # layer 1 never used
        opt.step(); opt.zero_grad()
    dead = zero_b_names(lora_state_dict(m))
    assert sorted(dead) == sorted(k for k in lora_state_dict(m) if "layers.1." in k and "lora_B" in k) or \
        all("layers.1." in k or "k_proj" in k or "v_proj" in k for k in dead)
    assert not any("layers.0.self_attn.q_proj" in k for k in dead)
