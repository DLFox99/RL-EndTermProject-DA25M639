#!/usr/bin/env python3
"""Synthetic test that Ctrl-C is archived as interrupted, not failed."""
import json
import tempfile
from pathlib import Path
from run_context import RunContext

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    model_dir = root / 'models' / 'nn_sarsa'
    model_dir.mkdir(parents=True)
    # Seed enough minimal metadata/log data for finalization to have progress.
    (model_dir / 'train_log.csv').write_text(
        'episode,timestep,episode_cost,wall_time_s,timestamp\n'
        '1,50,1000,0.1,2026-01-01T00:00:00\n', encoding='utf-8')
    (model_dir / 'training_metadata.json').write_text(
        json.dumps({'steps_completed': 50, 'episodes_completed': 1, 'wall_time_s': 0.1}),
        encoding='utf-8')
    cfg = {'techniques': {'nn_sarsa': {'num_episodes': 10}}}
    tc = {'portal_name': 'Neural Network based SARSA', 'category': 'nn_custom', 'num_episodes': 10}
    ctx = None
    try:
        with RunContext(technique='nn_sarsa', tech_config=tc, full_config=cfg,
                        model_dir=model_dir, force=True, run_name='interrupt-test',
                        repo_root=root, runs_root=root/'runs') as ctx:
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert ctx is not None
    summary = json.loads((ctx.run_dir/'summary.json').read_text())
    manifest = json.loads((ctx.run_dir/'manifest.json').read_text())
    assert summary['status'] == 'interrupted', summary
    assert manifest['status'] == 'interrupted', manifest
    assert manifest['failure']['type'] == 'KeyboardInterrupt', manifest
    print('PASS: interrupted run recorded as interrupted')
