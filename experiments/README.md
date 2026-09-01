# Controlled sweeps

Run a dry expansion first:

```bash
python3 sweep_runner.py experiments/nn_sarsa_epsilon_sweep.example.yaml --dry-run
```

Then run locally, optionally assigning CUDA devices:

```bash
python3 sweep_runner.py experiments/my_sweep.yaml --max-parallel 2 --devices 0,1
```

For multiple computers, use deterministic sharding. Example for three machines:

```bash
python3 sweep_runner.py experiments/my_sweep.yaml --shard-count 3 --shard-index 0
python3 sweep_runner.py experiments/my_sweep.yaml --shard-count 3 --shard-index 1
python3 sweep_runner.py experiments/my_sweep.yaml --shard-count 3 --shard-index 2
```

Each trial gets an isolated `models/` tree under `sweeps/<sweep>/<trial>/`, so
normal `models/<technique>` champion/DVC artifacts are never overwritten.

## Optional schedules

A technique may define `schedules:` in an experiment override. Supported scalar
schedule types are `constant`, `linear`, `cosine`, `exponential`, and
`piecewise`.

Example technique-local override:

```yaml
base_overrides:
  schedules:
    epsilon:
      type: cosine
      start: 1.0
      end: 0.02
      duration: 18000
```

Dynamic schedules are currently wired for:

- SB3 PPO/A2C/DQN/DDQN learning rate;
- SB3 DQN/DDQN epsilon/exploration rate;
- REINFORCE learning rate;
- tabular Q-learning/SARSA/TD(lambda) epsilon;
- NN Q-learning/SARSA epsilon and learning rate.

A3C remains sweepable through static hyperparameter overrides; its asynchronous
shared optimizer is intentionally not mutated by a parent-side dynamic schedule.
