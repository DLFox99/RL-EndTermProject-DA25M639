#!/usr/bin/env python3
"""Select, independently evaluate, assemble, and validate a portal candidate.

Example:
    python candidate_pipeline.py ppo \
        --sweep /path/to/ppo-shard-a \
        --sweep /path/to/ppo-shard-b \
        --top-k 3 \
        --batch ppo-stage2

Direct isolated runs are also supported with ``--model-dir /path/models/technique``.

The pipeline never writes into canonical ``models/<technique>``.  It ranks
completed sweep trials by their periodic deterministic ``best_eval_cost``,
re-evaluates the top K using the repository's independent evaluation set,
selects the lowest independent mean, packages that exact ``best_eval_model``,
validates only the requested technique, and writes an immutable manifest plus
hashes under ``portal_candidates/<batch>``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml

from evaluate import build_policy_fn, load_config
from evaluation_core import run_policy_evaluation
from industrial_inventory_env import generate_student_config

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "portal_candidates"

EXT_MAP = {
    "onpolicy": ".zip",
    "offpolicy": ".zip",
    "reinforce": ".pt",
    "a3c": ".pt",
    "nn_custom": ".pt",
    "tabular": ".npz",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_sweep_technique(sweep: Path) -> str | None:
    spec_path = sweep / "resolved_sweep.yaml"
    if not spec_path.exists():
        return None
    try:
        spec = yaml.safe_load(spec_path.read_text()) or {}
        return str(spec.get("technique")) if spec.get("technique") else None
    except Exception:
        return None


def _parse_overrides(text: str | None) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _checkpoint_path(model_dir: Path, tc: Dict[str, Any]) -> Path:
    ext = EXT_MAP.get(tc.get("category", "onpolicy"), ".zip")
    path = model_dir / f"best_eval_model{ext}"
    if not path.exists():
        raise FileNotFoundError(f"best_eval checkpoint missing: {path}")
    return path


def collect_candidates(
    sweeps: Iterable[Path], technique: str, tc: Dict[str, Any]
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for sweep_arg in sweeps:
        sweep = Path(sweep_arg).expanduser().resolve()
        summary = sweep / "sweep_summary.csv"
        if not summary.exists():
            raise FileNotFoundError(f"sweep summary not found: {summary}")
        declared = _load_sweep_technique(sweep)
        if declared is not None and declared != technique:
            raise ValueError(
                f"{sweep} declares technique={declared!r}, expected {technique!r}"
            )

        with summary.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rc = int(row.get("returncode", "1"))
                except Exception:
                    rc = 1
                if rc != 0:
                    continue
                try:
                    selection_cost = float(row["best_eval_cost"])
                except Exception:
                    continue
                if not math.isfinite(selection_cost):
                    continue

                trial_id = str(row["trial_id"])
                trial_dir = sweep / trial_id
                model_dir = trial_dir / "models" / technique
                try:
                    checkpoint = _checkpoint_path(model_dir, tc)
                except FileNotFoundError:
                    continue

                candidates.append({
                    "candidate_id": f"{sweep.name}:{trial_id}",
                    "trial_id": trial_id,
                    "sweep": str(sweep),
                    "trial_dir": str(trial_dir),
                    "models_root": str(trial_dir / "models"),
                    "model_dir": str(model_dir),
                    "checkpoint": str(checkpoint),
                    "selection_best_eval_cost": selection_cost,
                    "best_training_rolling_cost": row.get("best_training_rolling_cost"),
                    "steps_completed": row.get("steps_completed"),
                    "episodes_completed": row.get("episodes_completed"),
                    "overrides": _parse_overrides(row.get("overrides")),
                })

    if not candidates:
        raise RuntimeError("no successful candidates with best_eval checkpoints found")
    candidates.sort(key=lambda r: r["selection_best_eval_cost"])
    for rank, item in enumerate(candidates, start=1):
        item["selection_rank"] = rank
    return candidates


def collect_direct_candidates(
    model_dirs: Iterable[Path], technique: str, tc: Dict[str, Any]
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for model_arg in model_dirs:
        model_dir = Path(model_arg).expanduser().resolve()
        if model_dir.name != technique:
            raise ValueError(
                f"direct model directory must end in /{technique}: {model_dir}"
            )
        checkpoint = _checkpoint_path(model_dir, tc)
        evaluation_path = model_dir / "evaluation_metadata.json"
        if not evaluation_path.exists():
            raise FileNotFoundError(
                f"direct candidate lacks evaluation_metadata.json: {evaluation_path}"
            )
        evaluation = json.loads(evaluation_path.read_text())
        selection_cost = float(evaluation["best_mean_cost"])
        if not math.isfinite(selection_cost):
            raise ValueError(f"non-finite best_mean_cost in {evaluation_path}")

        training = {}
        training_path = model_dir / "training_metadata.json"
        if training_path.exists():
            try:
                training = json.loads(training_path.read_text())
            except Exception:
                training = {}

        params: Dict[str, Any] = {}
        params_path = model_dir / "hyperparams_used.yaml"
        if params_path.exists():
            try:
                value = yaml.safe_load(params_path.read_text()) or {}
                if isinstance(value, dict):
                    params = value
            except Exception:
                params = {}

        models_root = model_dir.parent
        run_root = models_root.parent
        candidates.append({
            "candidate_id": f"direct:{run_root.name}:{technique}",
            "trial_id": run_root.name,
            "sweep": "DIRECT",
            "trial_dir": str(run_root),
            "models_root": str(models_root),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint),
            "selection_best_eval_cost": selection_cost,
            "best_training_rolling_cost": training.get("best_rolling_cost"),
            "steps_completed": training.get("steps_completed"),
            "episodes_completed": training.get("episodes_completed"),
            "overrides": params,
        })
    return candidates


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k) for k in fields}
            if "overrides" in payload:
                payload["overrides"] = json.dumps(payload["overrides"], sort_keys=True)
            writer.writerow(payload)


def _evaluate_candidate(
    candidate: Dict[str, Any], technique: str, tc: Dict[str, Any],
    config: Dict[str, Any], student_config: Dict[str, Any], output_dir: Path,
) -> Dict[str, Any]:
    model_dir = Path(candidate["model_dir"])
    policy_fn = build_policy_fn(
        technique, tc, model_dir, checkpoint="best_eval"
    )
    result = run_policy_evaluation(
        policy_fn,
        student_config,
        seeds=config["evaluation"]["seeds"],
        scenario_modes=config["evaluation"]["scenario_modes"],
    )
    records = result["records"]
    costs = np.asarray([float(r["total_cost"]) for r in records], dtype=np.float64)
    mean = float(costs.mean())
    std = float(costs.std(ddof=1)) if len(costs) > 1 else 0.0

    eval_file = output_dir / "evaluations" / (
        f"rank{candidate['selection_rank']:02d}_{candidate['trial_id']}.csv"
    )
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    if records:
        fields = list(records[0].keys())
        with eval_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    return {
        **candidate,
        "independent_mean_cost": mean,
        "independent_std_cost": std,
        "independent_n_episodes": int(len(costs)),
        "evaluation_file": str(eval_file),
    }


def _copy_selected_metadata(winner: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    source_dir = output_dir / "selected_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}

    checkpoint = Path(winner["checkpoint"])
    dst = source_dir / checkpoint.name
    shutil.copy2(checkpoint, dst)
    copied["checkpoint"] = str(dst)

    model_dir = Path(winner["model_dir"])
    trial_dir = Path(winner["trial_dir"])
    for path in [
        model_dir / "hyperparams_used.yaml",
        model_dir / "training_metadata.json",
        model_dir / "evaluation_metadata.json",
        model_dir / "eval_metrics.csv",
        trial_dir / "trial.json",
    ]:
        if path.exists():
            dst = source_dir / path.name
            shutil.copy2(path, dst)
            copied[path.name] = str(dst)
    return copied


def main() -> int:
    config = load_config()
    techniques = list(config["techniques"].keys())

    parser = argparse.ArgumentParser(
        description="Select, independently evaluate, package, and validate a sweep candidate"
    )
    parser.add_argument("technique", choices=techniques)
    parser.add_argument("--sweep", action="append", default=[], type=Path,
                        help="Sweep directory; repeat for multiple shards/runs")
    parser.add_argument("--model-dir", action="append", default=[], type=Path,
                        help="Direct <...>/models/<technique> candidate; repeat as needed")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of selection-ranked candidates to independently evaluate")
    parser.add_argument("--batch", required=True,
                        help="Immutable batch name under portal_candidates/")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing batch directory")
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not args.sweep and not args.model_dir:
        raise ValueError("provide at least one --sweep or --model-dir source")

    technique = args.technique
    tc = config["techniques"][technique]
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.batch
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"candidate batch already exists: {output_dir}; use --force to replace"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    candidates: List[Dict[str, Any]] = []
    if args.sweep:
        candidates.extend(collect_candidates(args.sweep, technique, tc))
    if args.model_dir:
        candidates.extend(collect_direct_candidates(args.model_dir, technique, tc))
    if not candidates:
        raise RuntimeError("no usable candidates found")
    candidates.sort(key=lambda r: r["selection_best_eval_cost"])
    for rank, item in enumerate(candidates, start=1):
        item["selection_rank"] = rank

    ranking_fields = [
        "selection_rank", "candidate_id", "trial_id", "sweep",
        "selection_best_eval_cost", "best_training_rolling_cost",
        "steps_completed", "episodes_completed", "overrides",
    ]
    _write_csv(output_dir / "candidate_ranking.csv", candidates, ranking_fields)

    top = candidates[:min(args.top_k, len(candidates))]
    student_config = generate_student_config(config["roll_number"])
    evaluated: List[Dict[str, Any]] = []

    print(f"Technique: {technique}")
    print(f"Successful candidates: {len(candidates)}")
    print(f"Independent evaluation: top {len(top)}")
    for candidate in top:
        result = _evaluate_candidate(
            candidate, technique, tc, config, student_config, output_dir
        )
        evaluated.append(result)
        print(
            f"  selection #{result['selection_rank']:02d} {result['trial_id']} "
            f"selection={result['selection_best_eval_cost']:,.3f} -> "
            f"independent={result['independent_mean_cost']:,.3f} "
            f"± {result['independent_std_cost']:,.3f} "
            f"(n={result['independent_n_episodes']})"
        )

    evaluated.sort(key=lambda r: r["independent_mean_cost"])
    winner = evaluated[0]
    eval_fields = [
        "selection_rank", "candidate_id", "trial_id", "sweep",
        "selection_best_eval_cost", "independent_mean_cost",
        "independent_std_cost", "independent_n_episodes", "overrides",
        "evaluation_file",
    ]
    _write_csv(output_dir / "independent_evaluations.csv", evaluated, eval_fields)

    submissions_root = output_dir / "submissions"
    models_root = Path(winner["models_root"])
    assemble_cmd = [
        sys.executable, str(PROJECT_ROOT / "assemble.py"), technique,
        "--models-root", str(models_root),
        "--submissions-root", str(submissions_root),
        "--checkpoint", "best_eval",
    ]
    print("\nAssembling selected checkpoint...")
    subprocess.run(assemble_cmd, cwd=PROJECT_ROOT, check=True)

    validate_cmd = [
        sys.executable, str(PROJECT_ROOT / "validate.py"), technique,
        "--submissions-root", str(submissions_root),
    ]
    print("\nValidating selected portal artifact...")
    subprocess.run(validate_cmd, cwd=PROJECT_ROOT, check=True)

    copied = _copy_selected_metadata(winner, output_dir)
    upload_zip = submissions_root / technique / f"{technique}_upload.zip"
    selected_checkpoint = Path(copied["checkpoint"])

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "batch": args.batch,
        "technique": technique,
        "sweep_dirs": [str(Path(s).expanduser().resolve()) for s in args.sweep],
        "direct_model_dirs": [str(Path(s).expanduser().resolve()) for s in args.model_dir],
        "selection_metric": "periodic_deterministic_best_eval_cost",
        "independent_evaluation": {
            "seeds": config["evaluation"]["seeds"],
            "scenario_modes": config["evaluation"]["scenario_modes"],
            "n_episodes": winner["independent_n_episodes"],
        },
        "top_k": len(top),
        "winner": {
            "candidate_id": winner["candidate_id"],
            "trial_id": winner["trial_id"],
            "source_sweep": winner["sweep"],
            "overrides": winner["overrides"],
            "selection_rank": winner["selection_rank"],
            "selection_best_eval_cost": winner["selection_best_eval_cost"],
            "independent_mean_cost": winner["independent_mean_cost"],
            "independent_std_cost": winner["independent_std_cost"],
            "source_checkpoint": winner["checkpoint"],
            "source_checkpoint_sha256": _sha256(Path(winner["checkpoint"])),
        },
        "portal_artifact": {
            "path": str(upload_zip),
            "sha256": _sha256(upload_zip),
            "validated": True,
        },
        "portal_feedback": {
            "portal_score": None,
            "submitted_at_utc": None,
            "submitted_at_ist": None,
        },
        "evaluated_candidates": [
            {
                "selection_rank": r["selection_rank"],
                "candidate_id": r["candidate_id"],
                "trial_id": r["trial_id"],
                "selection_best_eval_cost": r["selection_best_eval_cost"],
                "independent_mean_cost": r["independent_mean_cost"],
                "independent_std_cost": r["independent_std_cost"],
                "overrides": r["overrides"],
            }
            for r in evaluated
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    hashes = [
        (upload_zip, _sha256(upload_zip)),
        (selected_checkpoint, _sha256(selected_checkpoint)),
        (manifest_path, _sha256(manifest_path)),
    ]
    with (output_dir / "SHA256SUMS.txt").open("w") as f:
        for path, digest in hashes:
            f.write(f"{digest}  {path.relative_to(output_dir)}\n")

    print("\n" + "=" * 72)
    print("READY TO UPLOAD")
    print("=" * 72)
    print(f"Technique          : {technique}")
    print(f"Winner             : {winner['candidate_id']}")
    print(f"Selection eval     : {winner['selection_best_eval_cost']:,.3f}")
    print(
        f"Independent eval   : {winner['independent_mean_cost']:,.3f} "
        f"± {winner['independent_std_cost']:,.3f} "
        f"(n={winner['independent_n_episodes']})"
    )
    print(f"Upload ZIP         : {upload_zip}")
    print(f"Upload SHA256      : {_sha256(upload_zip)}")
    print(f"Manifest           : {manifest_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
