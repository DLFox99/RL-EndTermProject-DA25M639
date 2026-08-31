#!/usr/bin/env python3
"""Assemble submission packages ready for portal upload.

Usage:
    python assemble.py            # assemble all techniques that have models
    python assemble.py ppo dqn    # assemble specific techniques
"""

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


def assemble_technique(tech_name, tc):
    """Generate policy.py, copy model, create upload zip."""
    model_dir = MODELS_DIR / tech_name
    sub_dir = SUBMISSIONS_DIR / tech_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    category = tc.get("category", "onpolicy")
    action_type = tc.get("action_type", "multi")
    portal_name = tc.get("portal_name", tech_name)

    # --- Determine model source (prefer best_model) ---
    if category == "reinforce":
        if (model_dir / "best_model.pt").exists():
            src_model = model_dir / "best_model.pt"
        elif (model_dir / "final_model.pt").exists():
            src_model = model_dir / "final_model.pt"
        else:
            print(f"  {tech_name}: no model found, skipping")
            return False
        model_artifact = f"{tech_name}_model.pt"
    else:
        if (model_dir / "best_model.zip").exists():
            src_model = model_dir / "best_model.zip"
        elif (model_dir / "final_model.zip").exists():
            src_model = model_dir / "final_model.zip"
        else:
            print(f"  {tech_name}: no model found, skipping")
            return False
        model_artifact = f"{tech_name}_model.zip"

    # model_name for SB3 load() — WITHOUT .zip extension
    model_name_for_load = model_artifact.replace(".zip", "") \
        if model_artifact.endswith(".zip") else model_artifact

    # --- Generate policy.py ---
    if category == "reinforce":
        code = POLICY_REINFORCE.format(
            model_name=model_artifact,
            hidden=tc.get("hidden", 128),
        )
    else:
        # Determine import
        algo = tc["algo"]
        if algo == "DoubleDQN":
            import_module = "stable_baselines3"
            import_class = "DQN"  # Load with DQN at inference
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

    if len(sys.argv) > 1:
        techniques = sys.argv[1:]
    else:
        techniques = list(config["techniques"].keys())

    print("Assembling submission packages...")
    count = 0
    for tech in techniques:
        if tech in config["techniques"]:
            if assemble_technique(tech, config["techniques"][tech]):
                count += 1
        else:
            print(f"  {tech}: not in config, skipping")

    print(f"\n{count} packages assembled in submissions/")
    print("Upload the *_upload.zip files to the portal.")


if __name__ == "__main__":
    main()
