# vla-lora-finetune

Does a small LoRA adapter, fine-tuned on one weak LIBERO-Spatial task, improve SmolVLA on that task compared with the pretrained checkpoint? **Answer from these runs: no measurable effect (null result), and a side check shows the adapters hurt at least one other task.** Every number below comes from a file under `results/`.

**Demo (pre-rendered rollouts, no model in the browser):** https://haiha-ux.github.io/vla-lora-finetune/ (the page in `docs/`). See "Demo" for what is and is not checked.

## Headline

- **Target task 5** (LIBERO-Spatial, "pick up the black bowl on the ramekin and place it on the plate"). Pretrained: 40/90 = 44.4%. LoRA trained with seed 0: 39/90 = 43.3%. LoRA trained with seed 1: 43/90 = 47.8%. Differences from pretrained: -1.1 points (95% bootstrap CI -15.6 to +13.3) and +3.3 points (CI -11.1 to +17.8). **Null result.** The intervals are about 15 points wide on each side, so a moderate effect in either direction is not excluded; only a large one is.
- **Other tasks (forgetting check, 10 episodes per task, one eval seed, tasks 1, 3 and 7 chosen in advance):** on task 7 both adapters are far below pretrained (1/10 and 0/10 against 7/10; -60 and -70 points, bootstrap CIs excluding zero). Task 3: 4/10 and 4/10 against 6/10 (within noise). Task 1: 10/10 and 5/10 against 5/10 (one adapter up, one level, inconsistent). So training on one task gave no gain there and hurt at least one other. I did not investigate why.
- The 2 adapters and 3 evaluation seeds are **not 6 independent samples**: both adapters share the same training data, hyper-parameters and the same 3 sets of evaluation layouts. Treat the per-adapter rows as two replicates of one experiment.

## Setup

- **Model:** `HuggingFaceVLA/smolvla_libero` (SmolVLA, rev `6721902`), 608.2M parameters counted by PyTorch (the earlier `libero-vla-repro` README quotes 450M from a different count; I did not reconcile them). Pretrained and fine-tuned runs go through the same `scripts/eval_lora.py` (lerobot 0.6.1 `lerobot-eval`, reused read-only from the sibling `libero-vla-repro` environment).
- **Why task 5:** in the earlier inference-only baseline (`libero-vla-repro`, 10 episodes per task, fixed init states) task 5 was the weakest task for all three runs: 4/10 for `HuggingFaceVLA/smolvla_libero`, 0/10 for `lerobot/smolvla_libero`, 1/10 for pi0-FAST. That is the selection step, so the baseline is **re-measured here on fresh seeds** (below) rather than reused, to avoid counting regression to the mean as an improvement. Re-measured pretrained success on task 5 is 44.4%, well above the 4/10 that got it picked, which is what that selection effect looks like.
- **Adapter:** hand-written LoRA (`scripts/lora.py`, no peft) on the `q, k, v, o` projections of the 32 text-model layers of the VLM (128 wrapped Linear layers), rank 16, alpha 32; the action expert, vision tower and everything else frozen. Trainable parameters 3,276,800 = 0.54% of 608,210,976 total. Adapter file 6.59 MB (`adapter.safetensors`, fp32) against the 1.22 GB checkpoint file (bf16). **Only 3,215,360 of the trainable parameters can change** (see "What didn't work", item 2).
- **Training data:** `aopolin-lv/libero_spatial_no_noops_lerobot_v21` (the 432 LIBERO-Spatial demonstrations, 52,970 frames, 362 MB download; whole suite fits, I used only the target task). Task 5 has 39 demonstrations (4,472 frames); 35 used for training (4,046 frames) and 4 held out for validation loss. 3,000 steps, batch 16 (about 12 epochs), AdamW lr 1e-4 with 100 warm-up steps and cosine decay, grad clip 1.0, TF32. About 27 minutes per run on this machine. Two training seeds (0 and 1) with the same split.
- **Normalisation:** the checkpoint's own mean/std statistics are applied (not recomputed from the demos), so training and evaluation use the same normalisation.
- **Evaluation:** task 5, 3 evaluation seeds (2001, 3001, 4001) x 30 episodes = 90 episodes per policy, identical seeds for pretrained and both adapters. Success rate with Wilson 95% interval on pooled episodes, per-seed values, and an episode-level bootstrap interval for differences (`scripts/summarize.py`).

### Differences from the earlier baseline (`libero-vla-repro`)

1. **`n_action_steps=10`** here instead of the checkpoint default of 1 (re-plan every step): about 11 s per episode instead of about 50 s. The earlier A/B in that repo found no visible difference on 2 tasks, but it was tiny. Both arms here use 10, so the comparison is fair, but absolute numbers are not comparable with the earlier 66% (100 episodes, n_action_steps=1) or 4/10 (task 5) figures.
2. **Fresh object placements** (`--env.init_states=false`, layouts sampled from the evaluation seed) instead of the 50 fixed LIBERO init states used earlier. The training demos were recorded from the standard LIBERO initial states; evaluating on other random placements of the same task is a slightly different distribution.
3. 30 episodes per seed per task instead of 10.

## Results

Task 5 (target), 90 episodes per policy (`results/summary.md`, `results/summary.json`, per-episode outcomes in `results/eval/`):

| Policy | Successes | Rate | 95% CI (Wilson) | Per eval seed 2001 / 3001 / 4001 | Mean of seeds |
|---|---|---|---|---|---|
| Pretrained | 40/90 | 44.4% | 34.6-54.7 | 15 / 11 / 14 of 30 | 44.4% |
| LoRA, train seed 0 | 39/90 | 43.3% | 33.6-53.6 | 14 / 11 / 14 of 30 | 43.3% |
| LoRA, train seed 1 | 43/90 | 47.8% | 37.8-58.0 | 12 / 12 / 19 of 30 | 47.8% |

Differences from pretrained on the same evaluation seeds:

| Comparison | Rate difference | 95% bootstrap CI | Per eval seed (2001 / 3001 / 4001) |
|---|---|---|---|
| LoRA seed 0 - pretrained | -1.1 points | -15.6 to +13.3 | -3 / 0 / 0 |
| LoRA seed 1 - pretrained | +3.3 points | -11.1 to +17.8 | -10 / +3 / +17 |

Note: per-seed differences are in percentage points of 30 episodes (one episode = 3.3 points; the values are the rounded summary output); the seed-1 spread (-10 to +17) is as large as the whole pooled effect, which is the behaviour to expect from noise.

Forgetting check (other tasks, **10 episodes each, one evaluation seed (2001)**, tasks chosen before looking at results; too few episodes to show anything small):

| Task | Pretrained | LoRA seed 0 | LoRA seed 1 | Difference seed 0 / seed 1 (bootstrap CI) |
|---|---|---|---|---|
| 1 | 5/10 | 10/10 | 5/10 | +50 (+20 to +80) / 0 (-40 to +40) |
| 3 | 6/10 | 4/10 | 4/10 | -20 (-60 to +20) / -20 (-60 to +20) |
| 7 | 7/10 | 1/10 | 0/10 | -60 (-90 to -20) / -70 (-100 to -40) |

Task 7 is the only clear signal in the repo and it is a loss. Task 1 for seed 0 is a gain of the same size in the other direction; with one adapter up and one level on that task, I do not read it as a real improvement.

Training loss (`results/train_lora_s*.json`): held-out validation loss (flow-matching loss on the 4 held-out episodes, fixed noise) went **up**, from 0.606 at step 0 to 0.643 (seed 0) and 0.638 (seed 1) at step 3000. Training-batch loss is noisy and flat around 0.5.

Latency (`results/timing_*.json`): median 0.158 s per action-chunk call pretrained against 0.162 s with the adapter (seed 2001, task 5); the adapter is not merged into the weights.

## Alignment checks (run before training)

Because a bad result could be a plumbing bug, `scripts/diagnose.py` and `scripts/crosscheck.py` check the pretrained checkpoint against the demonstrations (`results/diagnose.json`, `results/crosscheck.json`):

- **Gripper convention mismatch found and fixed.** The training dataset stores the gripper as 1 = open, 0 = close (OpenVLA convention); the checkpoint and simulator use -1 = open, +1 = close. Not converting raises the pretrained flow-matching loss on task-5 demos from 0.505 to 1.298. `scripts/svla.py::to_libero_gripper` converts (tested). After conversion actions and states are identical (max abs difference 0.0) to the first shard of `lerobot/libero_spatial_image`, which stores -1/+1.
- **Image order and orientation:** flipping images 180 degrees (0.600) or swapping the two cameras (0.642) both raise the loss over the aligned 0.505.
- Video decoding: AV1 video frames against the PNG frames from the other dataset differ by about 2 grey levels on average, and the checkpoint's loss is the same (0.318 vs 0.324).
- Dataset task order differs from the simulator's task order; mapped by instruction text (`DS_TO_LIBERO`, tested).
- **Adapter active at evaluation:** `eval_lora.py` checks the loaded adapter tensors equal the file, and prints the count of non-zero B matrices (126 of 128) in the run log.
- The aligned loss (0.5) is not small. I did not establish what the checkpoint's own training loss on these demos was, so I cannot say 0.5 is "normal"; the checks above show only that corruptions make it worse, not that nothing remains misaligned.

## What didn't work

1. **No gain on the target task**, and validation loss went up while training loss stayed flat. I did not tune learning rate, rank, steps, target layers or data (one configuration, chosen in advance). The checkpoint was probably already trained on these same demonstrations (the task-5 demos come from the standard LIBERO-Spatial set; I did not verify the checkpoint's training data), so extra fitting on 35 episodes may have had little left to add. That is a possible reading, not something I tested.
2. **Two of the 128 adapter layers can never train.** In both adapters the B matrices of `layers.31.self_attn.q_proj` and `layers.31.self_attn.o_proj` are exactly zero after training. The action expert only reads keys and values of the last VLM layer, so those two projections' outputs never reach the loss and get no gradient. My first evaluation script asserted all 128 B matrices were non-zero and aborted all 6 LoRA evaluations; the pipeline still printed `ALL_DONE` because a pipe hid the failing exit status. I fixed the assertion to compare the loaded tensors with the file and to allow exactly those two zeros (`tests/test_lora.py::test_unused_layer_keeps_zero_B_and_is_detected`), and made the pipeline stop on any failed step. The "3.28M trainable" count includes those two layers' 61,440 parameters; 3,215,360 actually change.
3. **The forgetting check did not run the first time**: my patch to `run_all.sh` missed its target line, the task list was empty, and the script again printed `ALL_DONE`. It ran in a second pass.
4. **Hurting other tasks.** Task 7 collapsed (1/10 and 0/10 against 7/10). I did not investigate the cause and do not propose a mechanism. Only one evaluation seed and 10 episodes were run per other task.
5. **Few training demonstrations**: 35 episodes (39 minus 4 held out). More data (e.g. the whole suite) was not tried; one run costs about 27 minutes.
6. Evaluation uses `n_action_steps=10` and fresh placements; neither was validated against the earlier baseline setting beyond what is stated above.
7. The weak-task choice was made from a single 10-episode baseline per task, which is exactly the selection risk; the fresh re-measurement (44.4%) shows task 5 was not as weak as it looked.

## Demo

`docs/index.html` is a static page (no framework, no model inference). It loads `docs/manifest.json` and plays `docs/rollouts/*.mp4`: real simulator episodes recorded during the evaluations above, seed 2001, **chosen by episode index (first 4 on task 5, first 3 on tasks 1, 3, 7), not by outcome**. Pick task, episode and two policies (default pretrained against LoRA seed 0) to see side-by-side rollouts with the success flag, plus the per-task table. The page states on its face that the rollouts are pre-rendered. Both policies get the same seed and therefore the same object layout. All four task-5 clips shown are failures for all three policies, which is representative of a 44% success rate only loosely; the recorded task-5 clips (first 10 episodes per run) have not been viewed beyond the manifest. Total tracked media 0.63 MB (H.264, 192 px, crf 34).

Checks done: served with `python3 -m http.server`, `index.html`, `manifest.json` and clips return 200 with the right content types, and the clips decode as 192x192 H.264. Also opened in Chromium: the page loads, clips play, and the task picker switches both clips.

## Reproduce

```bash
# environment: reuse libero-vla-repro's .venv (lerobot 0.6.1, see that README); scripts/env.sh points to it
python3 -c "from huggingface_hub import snapshot_download as d; d('aopolin-lv/libero_spatial_no_noops_lerobot_v21', repo_type='dataset', local_dir='data/libero_spatial')"
# optional cross-check shard
python3 -c "from huggingface_hub import snapshot_download as d; d('lerobot/libero_spatial_image', repo_type='dataset', local_dir='data/lerobot_spatial_image', allow_patterns=['meta/*','data/chunk-000/file-000.parquet'])"
source scripts/env.sh
heavy $PY scripts/diagnose.py 5 results/diagnose.json && heavy $PY scripts/crosscheck.py
setsid nohup scripts/run_all.sh > logs/run_all.log 2>&1 &     # evals, 2 trainings, LoRA evals, forgetting check; resumable
$PY scripts/summarize.py && $PY scripts/make_demo.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q tests
```

All heavy runs: `nice -n 19 ionice -c3`, 4 threads, `set_per_process_memory_fraction(0.3)`, one job at a time on a shared GB10 machine.

## Layout

- `scripts/lora.py` LoRA wrapper (inject, freeze, save/load, zero-B detection); `scripts/svla.py` data, normalisation, tokens, gripper conversion; `scripts/train.py`; `scripts/eval_lora.py` + `run_eval.sh` (lerobot-eval with the adapter attached); `scripts/diagnose.py`, `crosscheck.py`; `scripts/summarize.py`; `scripts/make_demo.py`; `scripts/run_all.sh`.
- `tests/` adapter injection, freezing, save/load round trip, unused-layer zeros, gripper conversion, Wilson/bootstrap.
- `results/` summaries, per-episode outcomes (`eval/`), raw lerobot-eval outputs (`eval_info/`), training logs (`train_lora_s*.json`, `.log`), diagnostics. `docs/` the demo.
- Not committed: `data/`, `ckpt/` (adapters, 6.6 MB each, regenerable), `logs/`.
