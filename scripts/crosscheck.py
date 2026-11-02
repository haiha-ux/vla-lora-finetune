"""Same frames from two sources of the same demos: the AV1-video dataset used for training (aopolin-lv) and
lerobot/libero_spatial_image (PNG in parquet, first parquet shard only). Checks decode + conventions agree."""
import io, json
import numpy as np, pandas as pd, torch
from PIL import Image
from svla import *

torch.cuda.set_per_process_memory_fraction(0.3)
stats, tok, pol = load_stats(), tokenizer(), load_policy().eval()
df = pd.read_parquet("data/lerobot_spatial_image/data/chunk-000/file-000.parquet")
ep = df[df.episode_index == df.episode_index.iloc[0]].reset_index(drop=True)   # episode 0 of both datasets
n = len(ep)
d = Demos({0})                              # dataset task 0 holds episode 0
sel = np.arange(0, n - 1, max(1, n // 32))[:32]
assert d.ep_id[0] == 0
mad = {}
png = {k: np.stack([np.array(Image.open(io.BytesIO(ep[c].iloc[i]["bytes"])).convert("RGB")) for i in sel])
       for k, c in [("observation.images.image", "observation.images.image"), ("observation.images.image2", "observation.images.wrist_image")]}
for k in png:
    mad[k] = float(np.abs(png[k].astype(float) - d.imgs[k][sel].astype(float)).mean())
act_png = np.stack(ep.action.to_numpy())[sel]
mad["action_max_abs_diff_after_gripper_conversion"] = float(np.abs(act_png - d.action[sel]).max())
mad["state_max_abs_diff"] = float(np.abs(np.stack(ep["observation.state"].to_numpy())[sel] - d.state[sel]).max())
g = torch.Generator(device="cuda").manual_seed(0)
noise, t = torch.randn(len(sel), CHUNK, 32, generator=g, device="cuda"), torch.rand(len(sel), generator=g, device="cuda") * 0.998 + 0.001
b = d.batch(sel, stats, tok)
with torch.no_grad():
    mad["loss_video_frames"] = pol(b, noise=noise, time=t)[0].item()
    for k in png:
        b[k] = torch.from_numpy(png[k]).permute(0, 3, 1, 2).float().cuda() / 255
    mad["loss_png_frames"] = pol(b, noise=noise, time=t)[0].item()
json.dump(mad, open("results/crosscheck.json", "w"), indent=1); print(mad)
