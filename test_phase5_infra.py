#!/usr/bin/env python3
"""Phase-5 overrides, schedules, and sweep-isolation tests."""

import json
import tempfile
from pathlib import Path

import yaml

from config_overrides import apply_overrides
from schedules import schedule_from_spec
from sweep_runner import _expand_trials


def test_overrides():
    cfg = {
        "seed": 42,
        "evaluation": {"plateau": {"patience": 4}},
        "techniques": {"nn_sarsa": {"epsilon_end": 0.05, "schedules": {}}},
    }
    out = apply_overrides(
        cfg,
        [
            "epsilon_end=0.02",
            'schedules.epsilon={"type":"linear","start":1.0,"end":0.02,"duration":100}',
            "evaluation.plateau.patience=6",
        ],
        technique="nn_sarsa",
    )
    assert cfg["techniques"]["nn_sarsa"]["epsilon_end"] == 0.05  # deep copy
    tc = out["techniques"]["nn_sarsa"]
    assert tc["epsilon_end"] == 0.02
    assert tc["schedules"]["epsilon"]["duration"] == 100
    assert out["evaluation"]["plateau"]["patience"] == 6


def test_schedules():
    linear = schedule_from_spec({"type": "linear", "start": 1, "end": 0, "duration": 10})
    assert linear.value(0) == 1.0
    assert abs(linear.value(5) - 0.5) < 1e-12
    assert linear.value(10) == 0.0
    assert linear.value(100) == 0.0

    cosine = schedule_from_spec({"type": "cosine", "start": 1, "end": 0, "duration": 10})
    assert cosine.value(0) == 1.0
    assert abs(cosine.value(10)) < 1e-12

    pw = schedule_from_spec({
        "type": "piecewise",
        "points": [[0, 1.0], [10, 0.5], [20, 0.1]],
    })
    assert pw.value(0) == 1.0
    assert abs(pw.value(5) - 0.75) < 1e-12
    assert abs(pw.value(20) - 0.1) < 1e-12
    assert abs(pw.value(100) - 0.1) < 1e-12


def test_grid_expansion():
    trials = _expand_trials({
        "base_overrides": {"device": "auto"},
        "grid": {"epsilon_end": [0.01, 0.02], "learning_rate": [1e-3, 3e-4]},
        "repeats": 2,
    })
    assert len(trials) == 8
    assert all(t["overrides"]["device"] == "auto" for t in trials)
    combos = {
        (t["overrides"]["epsilon_end"], t["overrides"]["learning_rate"])
        for t in trials
    }
    assert len(combos) == 4


if __name__ == "__main__":
    test_overrides()
    test_schedules()
    test_grid_expansion()
    print("PASS: Phase-5 typed overrides, scalar schedules, and sweep expansion")
