"""Build docs/rollouts/*.mp4 + docs/manifest.json from the recorded eval videos (seed 2001, first K episodes by index).

Episodes are chosen by index only, never by outcome; failures are kept. Re-encoded small (H.264, 192 px, crf 34).
"""
import json, os, subprocess
from summarize import load

SEED = 2001
TASKS = {1: "pick up the black bowl next to the ramekin and place it on the plate",
         3: "pick up the black bowl on the cookie box and place it on the plate",
         5: "pick up the black bowl on the ramekin and place it on the plate",
         7: "pick up the black bowl on the stove and place it on the plate"}
ARMS = {"pretrained": "Pretrained", "lora_s0": "LoRA (train seed 0)", "lora_s1": "LoRA (train seed 1)"}
K = {5: 4}   # episodes per task; others 3
os.makedirs("docs/rollouts", exist_ok=True)
runs = load()
man = {"seed": SEED, "tasks": {}, "arms": ARMS}
for t, text in TASKS.items():
    eps = []
    for e in range(K.get(t, 3)):
        item = {"episode": e, "arms": {}}
        for arm in ARMS:
            d = f"logs/eval_{arm}_s{SEED}" if t == 5 else f"logs/eval_{arm}_t{t}_s{SEED}"
            src = f"{d}/videos/libero_spatial_{t}/eval_episode_{e}.mp4"
            if not os.path.exists(src):
                continue
            ok = runs[(arm, t)][SEED][e]
            dst = f"docs/rollouts/{arm}_t{t}_e{e}.mp4"
            subprocess.run(["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-i", src, "-vf", "scale=192:192",
                            "-c:v", "libx264", "-crf", "34", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", dst], check=True)
            item["arms"][arm] = {"file": dst[5:], "success": bool(ok)}
        eps.append(item)
    man["tasks"][str(t)] = {"instruction": text, "episodes": eps}
summ = json.load(open("results/summary.json"))
man["table"] = [{"arm": k.split("|")[0], "task": int(k.split("|")[1][4:]), "successes": r["successes"], "episodes": r["episodes"],
                 "ci95": r["ci95_wilson"]} for k, r in summ["rows"].items()]
json.dump(man, open("docs/manifest.json", "w"), indent=1)
print(sum(os.path.getsize(f"docs/rollouts/{f}") for f in os.listdir("docs/rollouts")) / 1e6, "MB")
