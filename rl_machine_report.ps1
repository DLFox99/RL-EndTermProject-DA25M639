Write-Host "============================================================"
Write-Host " RL TRAINING MACHINE REPORT - WINDOWS"
Write-Host "============================================================"
Write-Host ""

Write-Host "===== BASIC SYSTEM ====="
Write-Host "ComputerName :" $env:COMPUTERNAME

$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor

Write-Host "OS           :" $os.Caption
Write-Host "Version      :" $os.Version
Write-Host "Manufacturer :" $cs.Manufacturer
Write-Host "Model        :" $cs.Model

Write-Host ""
Write-Host "===== CPU ====="
Write-Host "Name             :" $cpu.Name
Write-Host "Physical cores   :" $cpu.NumberOfCores
Write-Host "Logical CPUs     :" $cpu.NumberOfLogicalProcessors
Write-Host "Max MHz          :" $cpu.MaxClockSpeed

Write-Host ""
Write-Host "===== MEMORY ====="
$ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
Write-Host "Total RAM GB :" $ramGB

Write-Host ""
Write-Host "===== STORAGE ====="
Get-PhysicalDisk |
    Select-Object FriendlyName, MediaType,
                  @{Name="SizeGB";Expression={[math]::Round($_.Size/1GB,1)}} |
    Format-Table -AutoSize

Write-Host ""
Write-Host "===== GPU ====="
Get-CimInstance Win32_VideoController |
    Select-Object Name,
                  @{Name="VRAM_GB";Expression={[math]::Round($_.AdapterRAM/1GB,2)}},
                  DriverVersion |
    Format-Table -AutoSize

Write-Host ""
Write-Host "===== NVIDIA-SMI ====="

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi
} else {
    Write-Host "nvidia-smi not available"
}

Write-Host ""
Write-Host "===== GIT ====="

if (Get-Command git -ErrorAction SilentlyContinue) {
    git rev-parse --short HEAD 2>$null
    git branch --show-current 2>$null
    git status --short 2>$null
} else {
    Write-Host "git not available"
}

Write-Host ""
Write-Host "===== PYTHON / RL SOFTWARE / BENCHMARK ====="

@'
import sys
import time

print("Python       :", sys.version.replace("\n", " "))
print("Executable   :", sys.executable)

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
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}         :", p.name)
            print(f"GPU {i} VRAM    :", f"{p.total_memory/1024**3:.2f} GiB")
            print(f"GPU {i} CC      :", f"{p.major}.{p.minor}")

except Exception as e:
    print("PyTorch      : unavailable:", repr(e))
    torch = None

try:
    import stable_baselines3 as sb3
    print("SB3          :", sb3.__version__)
except Exception as e:
    print("SB3          : unavailable:", repr(e))

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

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        completed += 1

        if done:
            obs, info = env.reset()

    dt = time.perf_counter() - t0

    print("Steps tested :", f"{completed:,}")
    print("Elapsed      :", f"{dt:.3f} s")
    print("ENV STEPS/s  :", f"{completed/dt:,.0f}")

    env.close()

except Exception as e:
    print("Environment benchmark unavailable:")
    print(repr(e))

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

    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = ((y-target)**2).mean()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    iterations = 300
    t0 = time.perf_counter()

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = ((y-target)**2).mean()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0

    print(
        f"{str(device).upper():8s}: "
        f"{iterations/dt:,.1f} training updates/s "
        f"({dt:.3f} s)"
    )

benchmark_torch(torch.device("cpu"))

if torch.cuda.is_available():
    benchmark_torch(torch.device("cuda"))
'@ | python