"""lerobot-eval, unchanged, except that the loaded SmolVLA policy gets the LoRA adapter named by $LORA_ADAPTER.

With $LORA_ADAPTER unset it is the plain pretrained checkpoint (same code path, no adapter), so both arms of
the comparison run through this one file.  Also records per-chunk latency like libero-vla-repro/scripts/eval_timed.py.
"""
import json, os, sys, time
import torch

torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.3)
sys.argv = ["lerobot-eval"] + sys.argv[1:]

from safetensors import safe_open
from safetensors.torch import load_file
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lora import inject_lora, load_lora_state_dict, param_counts, zero_b_names

path = os.environ.get("LORA_ADAPTER")
_orig = SmolVLAPolicy.from_pretrained.__func__


def _patched(cls, *a, **k):
    pol = _orig(cls, *a, **k)
    if path:
        with safe_open(path, "pt") as f:
            meta = f.metadata()
        names = inject_lora(pol, r=int(meta["rank"]), alpha=float(meta["alpha"]))
        load_lora_state_dict(pol, load_file(path))
        pol.to("cuda")
        saved = load_file(path)
        loaded = {k: v.detach().cpu() for k, v in pol.state_dict().items() if "lora_" in k}
        assert all(torch.equal(loaded[k], saved[k].float()) for k in saved), "loaded adapter differs from the file"
        dead = zero_b_names(saved)
        assert all("layers.31.self_attn.q_proj" in k or "layers.31.self_attn.o_proj" in k for k in dead), f"unexpected zero B: {dead}"
        nonzero = len(names) - len(dead)
        print(f"[lora] active: {len(names)} wrapped layers, {nonzero} with non-zero B ({len(dead)} unused-layer zeros), {param_counts(pol)}", flush=True)
    else:
        print("[lora] none: pretrained checkpoint", flush=True)
    return pol


SmolVLAPolicy.from_pretrained = classmethod(_patched)
times = []
_get = SmolVLAPolicy._get_action_chunk


def _timed(*a, **k):
    torch.cuda.synchronize(); t = time.perf_counter()
    r = _get(*a, **k)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    return r


SmolVLAPolicy._get_action_chunk = _timed
from lerobot.scripts.lerobot_eval import main
try:
    main()
finally:
    if times:
        json.dump({"n_calls": len(times), "median_s": sorted(times)[len(times) // 2]}, open(os.environ.get("TIMING_OUT", "/dev/null"), "w"))
