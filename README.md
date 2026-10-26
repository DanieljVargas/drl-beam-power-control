# Energy-Aware Beam Power Control with Deep Reinforcement Learning

*A proof-of-concept prototype: a PPO agent learns to set per-sector beam power levels in a simulated 6G cell, trading energy consumption against service quality under time-varying traffic.*

Radio access networks consume most of their energy simply by staying on — a base station at zero load still draws over half of its full-load power. This prototype explores, in a deliberately simplified setting, whether a learned controller can exploit low-traffic periods better than the reactive rules used in practice: lowering or switching off sector beams when demand is low, while keeping unmet demand (service deficit) small.

## The environment

A custom [Gymnasium](https://gymnasium.farama.org/) environment (`src/env.py`) models one cell with **4 sectors** (N, S, E, W). Time is discrete: one simulated day = **96 steps** (15-minute slots).

- **Observation:** normalized traffic demand per sector (0–1).
- **Action:** per-sector beam power level — `Off` (0 Mbps, energy cost 0), `Low` (150 Mbps, cost 1.0), `High` (300 Mbps, cost 2.5).
- **Reward:** −(total energy cost) − 25 × (unmet demand) − 1 × (over-provisioned capacity). The weights make service deficit expensive and wasted capacity cheap-but-nonfree, so the agent must learn the energy/QoS tradeoff rather than simply switching everything off.

Four traffic scenarios are generated at reset:

| Scenario | Description |
|---|---|
| `standard` | Diurnal sinusoidal demand per sector, with phase offsets |
| `stadium_event` | Low ambient traffic; one sector spikes during a random event window |
| `night_fluctuations` | Mostly idle sectors with sparse, random bursts — the low-traffic regime |
| `mobility_shift` | A crowd drifts across sectors over the day (gradual hand-offs) |

## The agent and the baseline

- **Agent:** PPO ([Stable-Baselines3](https://stable-baselines3.readthedocs.io/)), MLP policy [128, 128], trained for 500,000 timesteps across 4 parallel environments (`src/train.py`).
- **Baseline:** the reactive rule a simple controller would use — read current demand and provision the smallest sufficient level (`Off` if demand is 0, `Low` up to 150 Mbps, `High` above). Because it reacts to the same step's demand and demand never exceeds `High` capacity, this baseline never incurs a deficit; it is a strong baseline, not a strawman.

Evaluation (`src/evaluate.py`) rolls out both controllers on identical seeded traffic traces for every scenario and reports energy cost, unmet demand, and energy saved.

## Results

Single evaluation seed (42); energy in relative cost units accumulated over one simulated day (96 steps × 4 sectors). Full step-by-step data: `results/evaluation_results.csv`.

| Scenario | Baseline energy | PPO energy | Energy saved | PPO deficit rate |
|---|---:|---:|---:|---:|
| standard | 631.0 | 624.5 | 1.0% | 0.22% |
| stadium_event | 424.5 | 414.5 | 2.4% | 0.48% |
| night_fluctuations | 225.0 | 175.0 | **22.2%** | 3.50% |
| mobility_shift | 484.5 | 443.0 | 8.6% | 1.65% |

**Reading the results honestly.** Against a reactive baseline with zero deficit by construction, the learned controller saves little during busy, predictable hours — there is little slack to exploit. The savings concentrate exactly where theory says they should: the mostly-idle `night_fluctuations` regime, where the agent saves **22.2%** of energy by aggressively powering sectors down between bursts. The cost is a 3.5% unmet-demand rate in that scenario: the agent accepts occasional deficits on unpredictable bursts in exchange for large idle-period savings. Quantifying and *bounding* that tradeoff — rather than letting a reward weight hide it — is the research question this prototype motivates.

## Interactive demo

A Streamlit dashboard (`app.py`) visualizes the trained agent sector by sector, scenario by scenario, against the baseline:

```bash
streamlit run app.py
```

## Repository structure

```
app.py                  Streamlit dashboard
src/
  env.py                Cellular6GEnv — sectorized beam power-control environment
  train.py              PPO training (500k timesteps)
  evaluate.py           Agent vs. heuristic baseline; exports the results CSV
  profiler.py           Inference-latency profiling of the trained policy
models/                 Trained PPO model (ppo_6g_model.zip)
experiments/            Earlier model variants (e.g. model_B)
tests/                  Environment tests
results/                Step-by-step evaluation output (evaluation_results.csv)
```

## How to run

```bash
pip install -r requirements.txt
python -m src.train        # train the PPO agent -> models/ppo_6g_model.zip
python -m src.evaluate     # evaluate vs. baseline, regenerate the CSV
streamlit run app.py       # interactive dashboard
python -m tests.test_agent # verify the trained model (energy / deficit / latency targets)
```

The verification script must be run as a module from the repository root, as shown above — invoking it by file path (`python tests/test_agent.py`) breaks its imports.

## Limitations and future work

This is a proof of concept, not a measurement study. The model is deliberately stylized: sectors with three discrete power levels stand in for antenna-level beamforming; there is no channel/SINR model, no propagation, and traffic is synthetic. Results are reported for a single evaluation seed; a multi-seed evaluation (mean ± std) is the obvious next step, followed by finer power levels, real traffic traces (e.g., the Telecom Italia Milan dataset), and an explicit constraint formulation that bounds worst-case access delay instead of pricing deficit in the reward.

The prototype motivates my planned graduate research: energy-efficient beamforming and sleep orchestration — jointly optimized, with bounded quality of service — for sparse, low-traffic networks where always-on operation wastes the most.

## Author

**Daniel José Vargas Ávila** — Telecommunications Engineer, Universidad Católica Andrés Bello (2025). Network Engineer / Administrator; previously RF Optimization Engineer Trainee at Digitel (Ericsson ENM, Huawei MAE, ZTE NetNumen).
