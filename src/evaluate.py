"""Evaluate the beamforming PPO agent against a rule-based heuristic baseline."""
import csv
import os
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from src.env import Cellular6GEnv

STEPS: int = 96
SEED: int = 42
MODEL_PATH: str = "models/ppo_6g_model.zip"
CSV_PATH: str = "results/evaluation_results.csv"

POWER_CAPACITY = Cellular6GEnv.POWER_CAPACITY
POWER_COST = Cellular6GEnv.POWER_COST
SECTOR_NAMES = Cellular6GEnv.SECTOR_NAMES
SCENARIOS = Cellular6GEnv.SCENARIOS


class HeuristicBaseline:
    """Traditional reactive rule-based beam controller.

    Per sector:
        traffic > 150  -> High (2)
        traffic > 0    -> Low  (1)
        traffic == 0   -> Off  (0)
    """

    def predict(self, traffic: np.ndarray) -> np.ndarray:
        action = np.zeros_like(traffic, dtype=np.int64)
        action[traffic > 0.0] = 1
        action[traffic > 150.0] = 2
        return action


def record_traffic_trace(
    env: Cellular6GEnv, scenario: str, steps: int, seed: int
) -> np.ndarray:
    """Record the raw per-sector traffic demand sequence for a seeded scenario."""
    env.reset(seed=seed, options={"scenario": scenario})
    trace = np.empty((steps, env.N_SECTORS), dtype=np.float32)
    noop = np.ones(env.N_SECTORS, dtype=np.int64)
    for t in range(steps):
        trace[t] = env.traffic_demand
        env.step(noop)
    return trace


def _rollout_agent(
    env: Cellular6GEnv, model: PPO, scenario: str, steps: int, seed: int
) -> np.ndarray:
    """Roll out the DRL agent and return the chosen power level per sector/step."""
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    actions = np.empty((steps, env.N_SECTORS), dtype=np.int64)
    for t in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        actions[t] = np.asarray(a, dtype=np.int64).reshape(env.N_SECTORS)
        obs, _r, _term, _trunc, _info = env.step(a)
    return actions


def _metrics(traffic: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    capacity = POWER_CAPACITY[actions]
    energy = float(np.sum(POWER_COST[actions]))
    deficit = float(np.sum(np.maximum(traffic - capacity, 0.0)))
    return {"energy": energy, "deficit": deficit}


def evaluate(model: PPO) -> Dict[str, dict]:
    """Run all scenarios, compare agent vs heuristic, and export a CSV."""
    env = Cellular6GEnv()
    baseline = HeuristicBaseline()

    results: Dict[str, dict] = {}
    rows: List[dict] = []

    for scenario in SCENARIOS:
        traffic = record_traffic_trace(env, scenario, STEPS, SEED)

        base_actions = np.stack([baseline.predict(traffic[t]) for t in range(STEPS)])
        agent_actions = _rollout_agent(env, model, scenario, STEPS, SEED)

        base_m = _metrics(traffic, base_actions)
        agent_m = _metrics(traffic, agent_actions)

        saved_pct = (
            (base_m["energy"] - agent_m["energy"]) / base_m["energy"] * 100.0
            if base_m["energy"] > 0
            else 0.0
        )

        total_demand = float(np.sum(traffic))
        results[scenario] = {
            "baseline_energy": base_m["energy"],
            "agent_energy": agent_m["energy"],
            "baseline_deficit": base_m["deficit"],
            "agent_deficit": agent_m["deficit"],
            "agent_deficit_rate_pct": (
                agent_m["deficit"] / total_demand * 100.0 if total_demand > 0 else 0.0
            ),
            "energy_saved_pct": saved_pct,
        }

        base_cap = POWER_CAPACITY[base_actions]
        agent_cap = POWER_CAPACITY[agent_actions]
        for t in range(STEPS):
            for s in range(env.N_SECTORS):
                rows.append(
                    {
                        "scenario": scenario,
                        "step": t,
                        "sector": SECTOR_NAMES[s],
                        "traffic_mbps": round(float(traffic[t, s]), 3),
                        "baseline_power": int(base_actions[t, s]),
                        "baseline_capacity": float(base_cap[t, s]),
                        "baseline_energy_cost": float(POWER_COST[base_actions[t, s]]),
                        "baseline_deficit": round(
                            float(max(traffic[t, s] - base_cap[t, s], 0.0)), 3
                        ),
                        "agent_power": int(agent_actions[t, s]),
                        "agent_capacity": float(agent_cap[t, s]),
                        "agent_energy_cost": float(POWER_COST[agent_actions[t, s]]),
                        "agent_deficit": round(
                            float(max(traffic[t, s] - agent_cap[t, s], 0.0)), 3
                        ),
                    }
                )

    _write_csv(rows, CSV_PATH)
    return results


def _write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_report(results: Dict[str, dict]) -> None:
    sep = "-" * 108
    hdr = (
        f"{'Scenario':<22}{'Base Energy':>13}{'Agent Energy':>14}"
        f"{'Base Deficit':>14}{'Agent Deficit':>15}{'Deficit Rate %':>16}{'Saved %':>14}"
    )
    print(sep)
    print(hdr)
    print(sep)
    for sc, m in results.items():
        print(
            f"{sc:<22}{m['baseline_energy']:>13.2f}{m['agent_energy']:>14.2f}"
            f"{m['baseline_deficit']:>14.2f}{m['agent_deficit']:>15.2f}"
            f"{m['agent_deficit_rate_pct']:>16.2f}{m['energy_saved_pct']:>14.2f}"
        )
    print(sep)
    print(f"Step-by-step results exported to: {CSV_PATH}")


def main() -> None:
    model = PPO.load(MODEL_PATH)
    results = evaluate(model)
    print_report(results)


if __name__ == "__main__":
    main()
