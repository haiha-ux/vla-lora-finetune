"""Collect logs/eval_<arm>_s<seed>/eval_info.json into results/eval/*.json and results/summary.{json,md}.

Arms: pretrained, lora_s0, lora_s1 (adapter trained with seed 0 / 1). Eval seeds are shared by all arms.
Stats: Wilson 95% interval on pooled episodes; paired-by-seed differences; episode-level bootstrap CI for
the pooled difference (episodes are independent layouts, so resampling episodes within each arm is fair).
"""
import glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(pattern="logs/eval_*_s*/eval_info.json"):
    """-> {(arm, task): {eval_seed: [0/1 per episode]}}"""
    runs = defaultdict(dict)
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"eval_(.+)_s(\d+)/", f)
        arm, seed = re.sub(r"_t\d+$", "", m.group(1)), int(m.group(2))
        for t in json.load(open(f))["per_task"]:
            runs[(arm, t["task_id"])][seed] = [int(bool(x)) for x in t["metrics"]["successes"]]
    return runs


def boot_diff(a, b, n=10000, seed=0):
    """95% CI of mean(b) - mean(a), resampling episodes of each arm independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]).tolist())


def row(eps_by_seed):
    allx = [x for s in sorted(eps_by_seed) for x in eps_by_seed[s]]
    k, n = sum(allx), len(allx)
    lo, hi = wilson(k, n)
    per = {str(s): sum(v) / len(v) for s, v in sorted(eps_by_seed.items())}
    return {"successes": k, "episodes": n, "rate": k / n, "ci95_wilson": [lo, hi], "per_seed_rate": per,
            "per_seed_successes": {str(s): sum(v) for s, v in sorted(eps_by_seed.items())},
            "mean_of_seeds": float(np.mean(list(per.values()))), "std_of_seeds": float(np.std(list(per.values()), ddof=1)) if len(per) > 1 else None}


def main():
    runs = load()
    os.makedirs("results/eval", exist_ok=True)
    out = {}
    for (arm, task), v in sorted(runs.items()):
        out[f"{arm}|task{task}"] = row(v)
        json.dump({"arm": arm, "task": task, "episodes_by_eval_seed": v}, open(f"results/eval/{arm}_task{task}.json", "w"))
    diffs = {}
    for task in sorted({t for _, t in runs}):
        base = runs.get(("pretrained", task))
        for arm in sorted({a for a, t in runs if t == task and a != "pretrained"}):
            if not base:
                continue
            ft = runs[(arm, task)]
            common = sorted(set(base) & set(ft))
            a = [x for s in common for x in base[s]]; b = [x for s in common for x in ft[s]]
            diffs[f"{arm}-pretrained|task{task}"] = {"diff": float(np.mean(b) - np.mean(a)), "ci95_bootstrap": boot_diff(a, b),
                                                      "per_seed_diff": {str(s): float(np.mean(ft[s]) - np.mean(base[s])) for s in common}}
    json.dump({"rows": out, "diffs": diffs}, open("results/summary.json", "w"), indent=1)
    L = ["| arm | task | successes | rate | 95% CI (Wilson) | per eval seed (successes/episodes) | mean of seeds |", "|---|---|---|---|---|---|---|"]
    for k, r in out.items():
        arm, task = k.split("|")
        ps = ", ".join(f"{s}: {r['per_seed_successes'][s]}/{len(runs[(arm, int(task[4:]))][int(s)])}" for s in r["per_seed_rate"])
        L.append(f"| {arm} | {task[4:]} | {r['successes']}/{r['episodes']} | {100*r['rate']:.1f}% | {100*r['ci95_wilson'][0]:.1f}-{100*r['ci95_wilson'][1]:.1f} | {ps} | {100*r['mean_of_seeds']:.1f}% |")
    L += ["", "| comparison | task | rate difference | 95% bootstrap CI | per eval seed |", "|---|---|---|---|---|"]
    for k, r in diffs.items():
        c, task = k.split("|")
        L.append(f"| {c} | {task[4:]} | {100*r['diff']:+.1f} pts | {100*r['ci95_bootstrap'][0]:+.1f} to {100*r['ci95_bootstrap'][1]:+.1f} | " +
                 ", ".join(f"{s}: {100*d:+.0f}" for s, d in r["per_seed_diff"].items()) + " |")
    open("results/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
