"""LoRA fine-tune SmolVLA on the demos of one LIBERO-Spatial task. Resumable (ckpt/<name>/last.pt).

usage: train.py <name> <libero_task> <seed> <steps> [batch] [rank]
"""
import json, os, sys, time
import numpy as np
import torch
from safetensors.torch import save_file
from lora import inject_lora, lora_state_dict, param_counts
from svla import *

name, task, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
bs = int(sys.argv[5]) if len(sys.argv) > 5 else 16
rank = int(sys.argv[6]) if len(sys.argv) > 6 else 16
LR, WARM, VAL_EPS, ALPHA = 1e-4, 100, 4, 2 * rank
out = f"ckpt/{name}"; os.makedirs(out, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.3); torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(seed); rng = np.random.default_rng(seed)

stats, tok = load_stats(), tokenizer()
pol = load_policy()
names = inject_lora(pol, r=rank, alpha=ALPHA)
pol.train()
counts = param_counts(pol)
print("lora layers", len(names), counts, flush=True)
params = [p for p in pol.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
sched = lambda s: min(1.0, (s + 1) / WARM) * (0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)) * 0.9 + 0.1)

d = Demos({LIBERO_TO_DS[task]})
eps = sorted(set(d.ep_id)); val_eps = set(np.random.default_rng(0).permutation(eps)[:VAL_EPS].tolist())  # same split for every seed
is_val = np.array([e in val_eps for e in d.ep_id])
train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
print(f"train frames {len(train_idx)} ({len(eps) - VAL_EPS} eps), val frames {len(val_idx)} ({VAL_EPS} eps)", flush=True)


def val_loss():
    pol.eval(); ls = []
    g = torch.Generator(device="cuda").manual_seed(123)
    for i in range(0, min(len(val_idx), 256), 16):
        b = d.batch(val_idx[i:i + 16], stats, tok)
        noise = torch.randn(len(b["action"]), CHUNK, 32, generator=g, device="cuda")
        t = torch.rand(len(b["action"]), generator=g, device="cuda") * 0.998 + 0.001
        with torch.no_grad():
            ls.append(pol(b, noise=noise, time=t)[0].item())
    pol.train(); return float(np.mean(ls))


step, log = 0, []
if os.path.exists(f"{out}/last.pt"):
    ck = torch.load(f"{out}/last.pt", map_location="cuda")
    sd = ck["lora"]; pol.load_state_dict({**pol.state_dict(), **{k: v.cuda() for k, v in sd.items()}})
    opt.load_state_dict(ck["opt"]); step, log = ck["step"], ck["log"]; rng = np.random.default_rng(seed + step)
    print("resumed at", step, flush=True)

if step == 0:
    log.append({"step": 0, "val_loss": val_loss()}); print(log[-1], flush=True)
t0 = time.time()
while step < steps:
    for g_ in opt.param_groups: g_["lr"] = LR * sched(step)
    b = d.batch(rng.choice(train_idx, bs, replace=False), stats, tok)
    loss = pol(b)[0]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True); step += 1
    if step % 20 == 0:
        log.append({"step": step, "loss": loss.item(), "s_per_step": (time.time() - t0) / 20}); t0 = time.time()
        if step % 100 == 0: print(log[-1], flush=True)
    if step % 250 == 0 or step == steps:
        log.append({"step": step, "val_loss": val_loss()}); print(log[-1], flush=True)
        torch.save({"lora": lora_state_dict(pol), "opt": opt.state_dict(), "step": step, "log": log}, f"{out}/last.pt.tmp")
        os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")

sd = lora_state_dict(pol)
save_file(sd, f"{out}/adapter.safetensors", metadata={"rank": str(rank), "alpha": str(ALPHA), "task": str(task), "seed": str(seed), "steps": str(steps)})
json.dump({"name": name, "libero_task": task, "seed": seed, "steps": steps, "batch": bs, "rank": rank, "alpha": ALPHA, "lr": LR,
           "target_layers": len(names), **counts, "adapter_bytes": os.path.getsize(f"{out}/adapter.safetensors"),
           "train_frames": int(len(train_idx)), "val_frames": int(len(val_idx)), "log": log}, open(f"results/train_{name}.json", "w"), indent=1)
print("done", flush=True)
