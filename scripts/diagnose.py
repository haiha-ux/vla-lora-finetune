"""Alignment checks on the PRETRAINED checkpoint against the demonstrations (run before training).

If keys, normalisation stats, gripper convention and image order are all right, the checkpoint's flow-matching
loss on the demos is low; each deliberate corruption below should make it clearly worse.
"""
import json, sys
import numpy as np
import torch
from svla import *

torch.cuda.set_per_process_memory_fraction(0.3)
torch.set_num_threads(4)
task = int(sys.argv[1]) if len(sys.argv) > 1 else 5      # LIBERO-Spatial task id
out = sys.argv[2] if len(sys.argv) > 2 else "results/diagnose.json"
stats, tok = load_stats(), tokenizer()
pol = load_policy().eval()
res = {"libero_task": task, "checkpoint": CKPT}


def loss_on(demos, idx_sets, noise_seed=0, **kw):
    ls = []
    for idx in idx_sets:
        b = demos.batch(idx, stats, tok, **kw)
        g = torch.Generator(device="cuda").manual_seed(noise_seed)
        noise = torch.randn(len(idx), CHUNK, 32, generator=g, device="cuda")
        t = torch.rand(len(idx), generator=g, device="cuda") * 0.998 + 0.001
        with torch.no_grad():
            ls.append(pol(b, noise=noise, time=t)[0].item())
    return float(np.mean(ls))


d = Demos({LIBERO_TO_DS[task]})
rng = np.random.default_rng(0)
idx_sets = [rng.choice(len(d), 16, replace=False) for _ in range(6)]
res["n_frames"], res["n_episodes"] = len(d), d.n_episodes
res["loss"] = {"as_trained (aligned)": loss_on(d, idx_sets)}
res["loss"]["images flipped 180deg"] = loss_on(d, idx_sets, flip_images=True)
res["loss"]["cameras swapped"] = loss_on(d, idx_sets, swap_cams=True)
d_raw = Demos({LIBERO_TO_DS[task]}, fix_gripper=False)
res["loss"]["gripper NOT converted"] = loss_on(d_raw, idx_sets)

# sampled chunk vs demonstration, normalised units, per action dim (first 10 steps of the chunk)
errs = []
for idx in idx_sets[:3]:
    b = d.batch(idx, stats, tok)
    pol.reset()
    with torch.no_grad():
        pred = pol._get_action_chunk({k: v for k, v in b.items() if k != "action"})[:, :10, :7]
    errs.append(((pred - b["action"][:, :10]) ** 2).mean((0, 1)).cpu().numpy())
res["sampled_chunk_mse_per_dim_normalised"] = np.mean(errs, 0).round(4).tolist()

# gripper sign: dataset uses 1 = open; finger opening (state[6]) is larger when open
g, f = d.action[:, 6], d.state[:, 6]   # g already converted: -1 = open
res["finger_state_mean_when_converted_gripper_minus1_(open)"] = float(f[g < 0].mean())
res["finger_state_mean_when_converted_gripper_plus1_(close)"] = float(f[g > 0].mean())
# dataset vs checkpoint statistics, full suite action/state ranges
A, S = torch.from_numpy(d.action), torch.from_numpy(d.state)
res["action_dim_mean_demo_vs_ckpt"] = [A.mean(0).numpy().round(3).tolist(), stats["action.mean"].numpy().round(3).tolist()]
res["state_dim_mean_demo_vs_ckpt"] = [S.mean(0).numpy().round(3).tolist(), stats["observation.state.mean"].numpy().round(3).tolist()]
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
