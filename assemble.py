#!/usr/bin/env python3
"""Assemble submission packages ready for portal upload.

Usage:
    python assemble.py            # assemble all techniques that have models
    python assemble.py ppo dqn    # assemble specific techniques
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# Policy file templates per action type
# These are self-contained — no external imports beyond standard + SB3/torch

POLICY_SB3_MULTI = '''"""Auto-generated policy for {portal_name}."""
import os
import numpy as np
from {import_module} import {import_class}

def _flatten(obs):
    inventory = np.asarray(obs["inventory"], dtype=np.float32) / 200.0
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    demand = np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0
    day = np.asarray(obs["day"], dtype=np.float32) / 50.0
    cap = np.asarray(obs["capacity_utilisation"], dtype=np.float32)
    return np.concatenate([inventory, pipeline, demand, day, cap])

_DIR = os.path.dirname(os.path.abspath(__file__))
_model = {import_class}.load(os.path.join(_DIR, "{model_name}"))

def run_policy(observation):
    flat = _flatten(observation)
    action, _ = _model.predict(flat, deterministic=True)
    return (np.asarray(action, dtype=np.int64) * 10).tolist()
'''

POLICY_SB3_DISCRETE = '''"""Auto-generated policy for {portal_name}."""
import os
import numpy as np
from {import_module} import {import_class}

def _flatten(obs):
    inventory = np.asarray(obs["inventory"], dtype=np.float32) / 200.0
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    demand = np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0
    day = np.asarray(obs["day"], dtype=np.float32) / 50.0
    cap = np.asarray(obs["capacity_utilisation"], dtype=np.float32)
    return np.concatenate([inventory, pipeline, demand, day, cap])

def _decode(action_int):
    a1 = action_int // 121
    a2 = (action_int % 121) // 11
    a3 = action_int % 11
    return [int(a1 * 10), int(a2 * 10), int(a3 * 10)]

_DIR = os.path.dirname(os.path.abspath(__file__))
_model = {import_class}.load(os.path.join(_DIR, "{model_name}"))

def run_policy(observation):
    flat = _flatten(observation)
    action, _ = _model.predict(flat, deterministic=True)
    return _decode(int(action))
'''

POLICY_REINFORCE = '''"""Auto-generated policy for REINFORCE."""
import os
import numpy as np
import torch
import torch.nn as nn

def _flatten(obs):
    inventory = np.asarray(obs["inventory"], dtype=np.float32) / 200.0
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    demand = np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0
    day = np.asarray(obs["day"], dtype=np.float32) / 50.0
    cap = np.asarray(obs["capacity_utilisation"], dtype=np.float32)
    return np.concatenate([inventory, pipeline, demand, day, cap])

class _ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim=38, hidden={hidden}, n_actions=11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(3)])
    def forward(self, x):
        h = self.shared(x)
        return [head(h) for head in self.heads]

_DIR = os.path.dirname(os.path.abspath(__file__))
_policy = _ReinforcePolicy()
_policy.load_state_dict(torch.load(
    os.path.join(_DIR, "{model_name}"), map_location="cpu", weights_only=True))
_policy.eval()

def run_policy(observation):
    flat = _flatten(observation)
    obs_t = torch.FloatTensor(flat).unsqueeze(0)
    with torch.no_grad():
        logits_list = _policy(obs_t)
    actions = [int(l.argmax(dim=-1).item()) for l in logits_list]
    return [a * 10 for a in actions]
'''


POLICY_TABULAR = '''"""Auto-generated policy for {portal_name}."""
import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

# Discretizer (must match training)
_INV_EDGES = np.array({inv_edges})
_CAP_EDGES = np.array({cap_edges})
_DAY_EDGES = np.array({day_edges})
_PIPE_EDGES = np.array({pipe_edges})
_N_INV = len(_INV_EDGES) + 1
_N_CAP = len(_CAP_EDGES) + 1
_N_DAY = len(_DAY_EDGES) + 1
_N_PIPE = len(_PIPE_EDGES) + 1

def _discretize(obs):
    inv = np.asarray(obs["inventory"], dtype=np.float64)
    cap = float(obs["capacity_utilisation"])
    day = float(obs["day"])
    pipe_sums = np.asarray(obs["arrival_pipeline"], dtype=np.float64).sum(axis=1)
    inv_bins = [int(np.digitize(inv[i], _INV_EDGES)) for i in range(3)]
    cap_bin = int(np.digitize(cap, _CAP_EDGES))
    day_bin = int(np.digitize(day, _DAY_EDGES))
    pipe_bins = [int(np.digitize(pipe_sums[i], _PIPE_EDGES)) for i in range(3)]
    idx = inv_bins[0]
    idx = idx * _N_INV + inv_bins[1]
    idx = idx * _N_INV + inv_bins[2]
    idx = idx * _N_CAP + cap_bin
    idx = idx * _N_DAY + day_bin
    idx = idx * _N_PIPE + pipe_bins[0]
    idx = idx * _N_PIPE + pipe_bins[1]
    idx = idx * _N_PIPE + pipe_bins[2]
    return idx

_data = np.load(os.path.join(_DIR, "{model_name}"))
_tables = [_data["q0"], _data["q1"], _data["q2"]]

def run_policy(observation):
    state = _discretize(observation)
    actions = [int(np.argmax(_tables[i][state])) for i in range(3)]
    return [a * 10 for a in actions]
'''

POLICY_NN_FACTORED = '''"""Auto-generated policy for {portal_name}."""
import os
import numpy as np
import torch
import torch.nn as nn

def _flatten(obs):
    inventory = np.asarray(obs["inventory"], dtype=np.float32) / 200.0
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    demand = np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0
    day = np.asarray(obs["day"], dtype=np.float32) / 50.0
    cap = np.asarray(obs["capacity_utilisation"], dtype=np.float32)
    return np.concatenate([inventory, pipeline, demand, day, cap])

class _QNet(nn.Module):
    def __init__(self, obs_dim=38, hidden={hidden}, n_actions=11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(3)])
    def forward(self, x):
        h = self.shared(x)
        return [head(h) for head in self.heads]

_DIR = os.path.dirname(os.path.abspath(__file__))
_model = _QNet()
_model.load_state_dict(torch.load(
    os.path.join(_DIR, "{model_name}"), map_location="cpu", weights_only=True))
_model.eval()

def run_policy(observation):
    flat = _flatten(observation)
    obs_t = torch.FloatTensor(flat).unsqueeze(0)
    with torch.no_grad():
        q_values = _model(obs_t)
    actions = [int(q.argmax(dim=-1).item()) for q in q_values]
    return [a * 10 for a in actions]
'''

POLICY_A3C = '''"""Auto-generated policy for A3C."""
import os
import numpy as np
import torch
import torch.nn as nn

def _flatten(obs):
    inventory = np.asarray(obs["inventory"], dtype=np.float32) / 200.0
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    demand = np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0
    day = np.asarray(obs["day"], dtype=np.float32) / 50.0
    cap = np.asarray(obs["capacity_utilisation"], dtype=np.float32)
    return np.concatenate([inventory, pipeline, demand, day, cap])

class _ActorCritic(nn.Module):
    def __init__(self, obs_dim=38, hidden={hidden}, n_actions=11):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(3)])
        self.critic = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return [head(h) for head in self.actor_heads], self.critic(h)

_DIR = os.path.dirname(os.path.abspath(__file__))
_model = _ActorCritic()
_model.load_state_dict(torch.load(
    os.path.join(_DIR, "{model_name}"), map_location="cpu", weights_only=True))
_model.eval()

def run_policy(observation):
    flat = _flatten(observation)
    obs_t = torch.FloatTensor(flat).unsqueeze(0)
    with torch.no_grad():
        logits, _ = _model(obs_t)
    actions = [int(l.argmax(dim=-1).item()) for l in logits]
    return [a * 10 for a in actions]
'''


def assemble_technique(tech_name, tc, models_root=MODELS_DIR,
                       submissions_root=SUBMISSIONS_DIR, checkpoint="auto"):
    """Generate policy.py, copy model, create upload zip.

    ``checkpoint='auto'`` prefers deterministic-evaluation selection, then
    the training-rolling best, then the final checkpoint.
    """
    model_dir = Path(models_root) / tech_name
    sub_dir = Path(submissions_root) / tech_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    category = tc.get("category", "onpolicy")
    action_type = tc.get("action_type", "multi")
    portal_name = tc.get("portal_name", tech_name)

    # --- Determine model source ---
    ext_map = {
        "onpolicy": ".zip", "offpolicy": ".zip",
        "reinforce": ".pt", "a3c": ".pt",
        "nn_custom": ".pt", "tabular": ".npz",
    }
    ext = ext_map.get(category, ".zip")

    checkpoint_names = {
        "best_eval": [f"best_eval_model{ext}"],
        "best_train": [f"best_model{ext}"],
        "final": [f"final_model{ext}"],
        "auto": [f"best_eval_model{ext}", f"best_model{ext}", f"final_model{ext}"],
    }
    if checkpoint not in checkpoint_names:
        raise ValueError(f"unknown checkpoint selector: {checkpoint}")

    src_model = None
    for name in checkpoint_names[checkpoint]:
        candidate = model_dir / name
        if candidate.exists():
            src_model = candidate
            break
    if src_model is None:
        print(f"  {tech_name}: no {checkpoint} checkpoint found in {model_dir}, skipping")
        return False

    model_artifact = f"{tech_name}_model{ext}"

    # model_name for SB3 load() — WITHOUT .zip extension
    model_name_for_load = model_artifact.replace(".zip", "") \
        if model_artifact.endswith(".zip") else model_artifact

    # --- Generate policy.py ---
    if category == "tabular":
        # Load discretizer config
        disc_cfg_path = model_dir / "discretizer_config.json"
        if disc_cfg_path.exists():
            import json as _json
            with open(disc_cfg_path) as f:
                disc_cfg = _json.load(f)
        else:
            # Defaults matching StateDiscretizer
            disc_cfg = {"inv_edges": [30, 60, 100, 150],
                        "cap_edges": [0.3, 0.6, 0.85],
                        "day_edges": [12, 25, 38],
                        "pipe_edges": [30, 80]}
        code = POLICY_TABULAR.format(
            portal_name=portal_name,
            model_name=model_artifact,
            inv_edges=disc_cfg["inv_edges"],
            cap_edges=disc_cfg["cap_edges"],
            day_edges=disc_cfg["day_edges"],
            pipe_edges=disc_cfg["pipe_edges"],
        )
    elif category in ("reinforce", "nn_custom"):
        hidden = tc.get("hidden", 128)
        if category == "nn_custom":
            code = POLICY_NN_FACTORED.format(
                portal_name=portal_name, model_name=model_artifact, hidden=hidden)
        else:
            code = POLICY_REINFORCE.format(
                model_name=model_artifact, hidden=hidden)
    elif category == "a3c":
        code = POLICY_A3C.format(
            model_name=model_artifact, hidden=tc.get("hidden", 128))
    else:
        # SB3 models
        algo = tc["algo"]
        if algo == "DoubleDQN":
            import_module = "stable_baselines3"
            import_class = "DQN"
        else:
            import_module = "stable_baselines3"
            import_class = algo

        template = POLICY_SB3_DISCRETE if action_type == "discrete" \
            else POLICY_SB3_MULTI
        code = template.format(
            portal_name=portal_name,
            import_module=import_module,
            import_class=import_class,
            model_name=model_name_for_load,
        )

    # Write policy file (named per technique for git tracking)
    policy_path = sub_dir / f"policy_{tech_name}.py"
    with open(policy_path, "w") as f:
        f.write(code)

    # --- Copy model artifact ---
    dst_model = sub_dir / model_artifact
    shutil.copy2(str(src_model), str(dst_model))

    # --- Create upload zip (policy.py + model artifact at root) ---
    upload_zip = sub_dir / f"{tech_name}_upload.zip"
    with zipfile.ZipFile(str(upload_zip), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(policy_path), "policy.py")
        zf.write(str(dst_model), model_artifact)

    print(f"  {tech_name}: {upload_zip.name} ready "
          f"(policy.py + {model_artifact})")
    return True


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Assemble portal submission packages")
    parser.add_argument("techniques", nargs="*",
                        help="Techniques to assemble; default: all configured techniques")
    parser.add_argument("--models-root", type=Path, default=MODELS_DIR,
                        help="Root containing <technique>/ checkpoint directories")
    parser.add_argument("--submissions-root", type=Path, default=SUBMISSIONS_DIR,
                        help="Output root for assembled submission directories")
    parser.add_argument(
        "--checkpoint",
        choices=["auto", "best_eval", "best_train", "final"],
        default="auto",
        help="Checkpoint selection; auto prefers best_eval, then best_train, then final",
    )
    args = parser.parse_args()

    techniques = args.techniques or list(config["techniques"].keys())
    models_root = args.models_root.expanduser().resolve()
    submissions_root = args.submissions_root.expanduser().resolve()

    print("Assembling submission packages...")
    count = 0
    for tech in techniques:
        if tech in config["techniques"]:
            if assemble_technique(
                tech, config["techniques"][tech],
                models_root=models_root,
                submissions_root=submissions_root,
                checkpoint=args.checkpoint,
            ):
                count += 1
        else:
            print(f"  {tech}: not in config, skipping")

    print(f"\n{count} packages assembled in {submissions_root}/")
    print("Upload the *_upload.zip files to the portal.")


if __name__ == "__main__":
    main()
