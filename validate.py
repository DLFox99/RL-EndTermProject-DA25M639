#!/usr/bin/env python3
"""Validate all submission policies against the official test script.

Usage:
    python validate.py           # validate all
    python validate.py ppo dqn   # validate specific techniques
"""

import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if len(sys.argv) > 1:
        techniques = sys.argv[1:]
    else:
        techniques = list(config["techniques"].keys())

    results = {}
    for tech in techniques:
        policy_file = SUBMISSIONS_DIR / tech / f"policy_{tech}.py"
        if not policy_file.exists():
            print(f"  {tech}: policy file not found — run assemble first")
            results[tech] = "MISSING"
            continue

        print(f"\n--- Validating {tech} ---")
        ret = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "policy_validation_tests.py"),
             str(policy_file)],
            capture_output=False,
        )
        results[tech] = "PASSED" if ret.returncode == 0 else "FAILED"

    print(f"\n{'='*40}")
    print("  Validation Summary")
    print(f"{'='*40}")
    for tech, status in results.items():
        icon = "✓" if status == "PASSED" else "✗"
        print(f"  {icon}  {tech}: {status}")

    failed = [t for t, s in results.items() if s != "PASSED"]
    if failed:
        print(f"\n{len(failed)} FAILED. Fix before uploading.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} policies passed.")


if __name__ == "__main__":
    main()
