"""Shared pieces: policy loading, checkpoint normalisation stats, language tokens, demo dataset.

Everything that touches observation/action keys lives here so train, diagnose and eval agree.
"""
import glob
import json
import os

import av
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

CKPT = "HuggingFaceVLA/smolvla_libero"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "libero_spatial")
# dataset task_index -> LIBERO-Spatial task id (matched by instruction text; the two orders differ)
DS_TO_LIBERO = {0: 6, 1: 4, 2: 5, 3: 7, 4: 0, 5: 3, 6: 8, 7: 1, 8: 2, 9: 9}
LIBERO_TO_DS = {v: k for k, v in DS_TO_LIBERO.items()}
CAMS = {"observation.images.image": "image", "observation.images.image2": "wrist_image"}  # policy key -> dataset video dir
CHUNK = 50


def ckpt_dir():
    return glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/*"))[0]


def load_stats():
    sd = load_file(os.path.join(ckpt_dir(), "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    return {k: v.float() for k, v in sd.items()}


def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")


def tokenize(tok, texts, max_len=48):
    """Same as the checkpoint's preprocessor: instruction + newline, right-padded to the longest, max 48."""
    out = tok([t if t.endswith("\n") else t + "\n" for t in texts], max_length=max_len, padding="longest",
              truncation=True, padding_side="right", return_tensors="pt")
    return out["input_ids"], out["attention_mask"].bool()


def load_policy(n_action_steps=10, device="cuda"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    p = SmolVLAPolicy.from_pretrained(CKPT)
    p.config.n_action_steps = n_action_steps
    return p.to(device)


def to_libero_gripper(action):
    """This dataset uses the OpenVLA gripper convention (1 = open, 0 = close). The checkpoint (and the simulator)
    use LIBERO's raw convention (-1 = open, +1 = close), so convert: raw = 1 - 2 * g."""
    a = np.array(action, dtype=np.float32, copy=True)
    a[..., 6] = 1.0 - 2.0 * a[..., 6]
    return a


def _decode(path):
    c = av.open(path)
    fr = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    c.close()
    return fr


class Demos:
    """All demonstrations of the given dataset tasks, held in RAM (uint8 images)."""

    def __init__(self, ds_tasks, root=DATA, fix_gripper=True):
        eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl")]
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{root}/meta/tasks.jsonl")}
        self.imgs = {k: [] for k in CAMS}
        self.state, self.action, self.ep_end, self.text, self.ep_id = [], [], [], [], []
        n = 0
        for e in eps:
            df = pd.read_parquet(f"{root}/data/chunk-000/episode_{e['episode_index']:06d}.parquet")
            if int(df.task_index.iloc[0]) not in ds_tasks:
                continue
            for k, d in CAMS.items():
                fr = _decode(f"{root}/videos/chunk-000/observation.images.{d}/episode_{e['episode_index']:06d}.mp4")
                assert len(fr) == len(df), (e, len(fr), len(df))
                self.imgs[k].append(fr)
            self.state.append(np.stack(df["observation.state"].to_numpy()))
            self.action.append(np.stack(df["action"].to_numpy()))
            n += len(df)
            self.ep_end += [n] * len(df)
            self.text += [tasks[int(df.task_index.iloc[0])]] * len(df)
            self.ep_id += [e["episode_index"]] * len(df)
        self.imgs = {k: np.concatenate(v) for k, v in self.imgs.items()}
        self.state = np.concatenate(self.state).astype(np.float32)
        self.action = to_libero_gripper(np.concatenate(self.action)) if fix_gripper else np.concatenate(self.action).astype(np.float32)
        self.ep_end = np.array(self.ep_end)
        self.n_episodes = len(set(self.ep_id))

    def __len__(self):
        return len(self.state)

    def batch(self, idx, stats, tok, device="cuda", flip_images=False, swap_cams=False):
        """Model-ready batch: images [0,1], normalised state, normalised 50-step action chunk (+pad mask), tokens."""
        idx = np.asarray(idx)
        steps = idx[:, None] + np.arange(CHUNK)[None]
        pad = steps >= self.ep_end[idx][:, None]
        steps = np.minimum(steps, self.ep_end[idx][:, None] - 1)
        act = torch.from_numpy(self.action[steps])
        b = {}
        keys = list(CAMS)[::-1] if swap_cams else list(CAMS)
        for pk, dk in zip(CAMS, keys):
            im = torch.from_numpy(self.imgs[dk][idx]).permute(0, 3, 1, 2).float() / 255
            b[pk] = im.flip(-1, -2) if flip_images else im
        b["observation.state"] = normalize(torch.from_numpy(self.state[idx]), stats["observation.state.mean"], stats["observation.state.std"])
        b["action"] = normalize(act, stats["action.mean"], stats["action.std"])
        b["action_is_pad"] = torch.from_numpy(pad)
        b["observation.language.tokens"], b["observation.language.attention_mask"] = tokenize(tok, [self.text[i] for i in idx])
        return {k: v.to(device) for k, v in b.items()}
