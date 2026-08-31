#!/usr/bin/env bash

set -u

echo "============================================================"
echo " RL TRAINING MACHINE REPORT"
echo "============================================================"
echo

echo "===== BASIC SYSTEM ====="
echo "Hostname     : $(hostname)"
echo "Date         : $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Kernel       : $(uname -r)"
echo "Architecture : $(uname -m)"

if command -v hostnamectl >/dev/null 2>&1; then
    hostnamectl 2>/dev/null | grep -E \
        'Operating System|Hardware Vendor|Hardware Model' || true
fi

echo
echo "===== CPU ====="

if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E \
        'Model name:|Socket\(s\):|Core\(s\) per socket:|Thread\(s\) per core:|CPU\(s\):|CPU max MHz:|CPU min MHz:|L3 cache:'
else
    cat /proc/cpuinfo | grep 'model name' | head -1
fi

echo
echo "===== MEMORY ====="
free -h

echo
echo "===== STORAGE ====="

if command -v lsblk >/dev/null 2>&1; then
    lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN
fi

echo
echo "===== PCI GPUs ====="

if command -v lspci >/dev/null 2>&1; then
    lspci | grep -Ei 'VGA|3D|Display' || true
fi

echo
echo "===== NVIDIA ====="

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=name,memory.total,driver_version,pstate,temperature.gpu \
        --format=csv,noheader
else
    echo "nvidia-smi not available"
fi

echo
echo "===== AMD ROCm ====="

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -E \
        'Name:|Marketing Name:' | head -20
else
    echo "rocminfo not available"
fi

echo
echo "===== REPOSITORY ====="

if command -v git >/dev/null 2>&1 &&
   git rev-parse --is-inside-work-tree >/dev/null 2>&1; then

    echo "Commit : $(git rev-parse --short HEAD)"
    echo "Branch : $(git branch --show-current)"

    if git diff --quiet &&
       git diff --cached --quiet; then
        echo "Tree   : clean"
    else
        echo "Tree   : MODIFIED"
    fi
else
    echo "Not running inside a Git repository"
fi

echo
echo "===== PYTHON / RL SOFTWARE / BENCHMARK ====="

python3 <<'PY'
import os
import platform
import subprocess
import sys
import time

print("Python       :", sys.version.replace("\n", " "))
print("Executable   :", sys.executable)

# ------------------------------------------------------------
# Package versions
# ------------------------------------------------------------

try:
    import numpy as np
    print("NumPy        :", np.__version__)
except Exception as e:
    print("NumPy        : unavailable:", repr(e))

try:
    import torch

    print("PyTorch      :", torch.__version__)
    print("Torch threads:", torch.get_num_threads())
    print("CUDA build   :", torch.version.cuda)
    print("CUDA usable  :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA GPUs    :", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)

            print(
                f"GPU {i}         : {prop.name}"
            )
            print(
                f"GPU {i} VRAM    : "
                f"{prop.total_memory / 1024**3:.2f} GiB"
            )
            print(
                f"GPU {i} CC      : "
                f"{prop.major}.{prop.minor}"
            )

except Exception as e:
    print("PyTorch      : unavailable:", repr(e))
    torch = None

try:
    import stable_baselines3 as sb3
    print("SB3          :", sb3.__version__)
except Exception as e:
    print("SB3          : unavailable:", repr(e))


# ------------------------------------------------------------
# Industrial Inventory Environment benchmark
# ------------------------------------------------------------

print()
print("----- INVENTORY ENVIRONMENT BENCHMARK -----")

try:
    import yaml
    from industrial_inventory_env import (
        IndustrialInventoryEnv,
        generate_student_config,
    )

    roll_number = "DA25M639"

    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
            roll_number = cfg.get("roll_number", roll_number)
    except Exception:
        pass

    student_config = generate_student_config(roll_number)

    env = IndustrialInventoryEnv(
        student_config,
        scenario_mode="random",
        domain_randomization=True,
    )

    obs, info = env.reset(seed=12345)

    N = 50000
    completed = 0

    t0 = time.perf_counter()

    for _ in range(N):
        action = env.action_space.sample()

        result = env.step(action)

        # Gymnasium API
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        completed += 1

        if done:
            obs, info = env.reset()

    dt = time.perf_counter() - t0

    print(f"Steps tested : {completed:,}")
    print(f"Elapsed      : {dt:.3f} s")
    print(f"ENV STEPS/s  : {completed / dt:,.0f}")

    env.close()

except Exception as e:
    print("Environment benchmark unavailable:")
    print(repr(e))


# ------------------------------------------------------------
# PyTorch MLP benchmark
# ------------------------------------------------------------

print()
print("----- PYTORCH MLP TRAINING BENCHMARK -----")

if torch is None:
    print("Skipped: PyTorch unavailable")
    raise SystemExit


def benchmark_torch(device):
    torch.manual_seed(1234)

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    x = torch.randn(256, 128, device=device)
    target = torch.randn(256, 64, device=device)

    # Warmup
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    iterations = 300

    t0 = time.perf_counter()

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0

    print(
        f"{str(device).upper():8s}: "
        f"{iterations / dt:,.1f} training updates/s "
        f"({dt:.3f} s)"
    )


try:
    benchmark_torch(torch.device("cpu"))
except Exception as e:
    print("CPU benchmark failed:", repr(e))

if torch.cuda.is_available():
    try:
        benchmark_torch(torch.device("cuda"))
    except Exception as e:
        print("CUDA benchmark failed:", repr(e))

print()
print("============================================================")
PY
