#!/usr/bin/env python3
"""Environment sanity check. Run before training on any new machine.

Usage:
    python check_env.py
"""

import subprocess
import sys


def check(name, fn):
    print(f"  {name:.<50}", end=" ", flush=True)
    try:
        result = fn()
        print(f"OK  {result if result else ''}")
        return True
    except Exception as e:
        print(f"FAIL  ({e})")
        return False


def check_torch_cpu_only():
    import torch
    if torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is True — GPU build detected. "
            "This project should use CPU-only torch (see pipeline notes)."
        )
    return f"torch {torch.__version__}, CUDA disabled"


def check_no_triton():
    try:
        import triton  # noqa
        raise RuntimeError(
            "triton is importable — known cause of segfaults with CPU torch. "
            "Run: pip uninstall triton -y  (and remove any ~/.local copy)"
        )
    except ModuleNotFoundError:
        return "not installed (expected)"


def check_sb3_import():
    import stable_baselines3
    return f"stable-baselines3 {stable_baselines3.__version__}"


def check_gymnasium_import():
    import gymnasium
    return f"gymnasium {gymnasium.__version__}"


def check_env_package():
    from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config  # noqa
    return "industrial_inventory_env importable"


def check_yaml():
    import yaml  # noqa
    return "pyyaml available"


def check_ppo_smoke_test():
    """Actually build and train a PPO model for a handful of steps.
    This is the real test — import checks alone won't catch the
    segfault-on-optimizer-init issue we hit before.
    """
    from stable_baselines3 import PPO
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, device="cpu", verbose=0)
    model.learn(500)
    return "PPO build + short train succeeded"


def check_config_loads():
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    n = len(config.get("techniques", {}))
    return f"config.yaml OK, {n} techniques defined"


def check_student_config():
    from industrial_inventory_env import generate_student_config
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    sc = generate_student_config(config["roll_number"])
    return f"roll {config['roll_number']} → variant generated"


CHECKS = [
    ("PyTorch is CPU-only (no CUDA)",     check_torch_cpu_only),
    ("triton NOT importable",             check_no_triton),
    ("stable-baselines3 importable",      check_sb3_import),
    ("gymnasium importable",              check_gymnasium_import),
    ("industrial_inventory_env importable", check_env_package),
    ("pyyaml importable",                 check_yaml),
    ("config.yaml loads",                 check_config_loads),
    ("student config generates",          check_student_config),
    ("PPO smoke test (real train call)",  check_ppo_smoke_test),
]


def main():
    print("\nEnvironment Doctor")
    print("=" * 60)

    results = [check(name, fn) for name, fn in CHECKS]

    print("=" * 60)
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"All {total} checks passed. Safe to train.\n")
        sys.exit(0)
    else:
        print(f"{passed}/{total} checks passed. Fix the FAIL items above "
              f"before running training — a segfault mid-training wastes "
              f"far more time than fixing this now.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
