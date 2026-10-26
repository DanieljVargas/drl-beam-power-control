"""Computational-complexity profiler: heuristic vs PPO inference latency."""
import time

import numpy as np
from stable_baselines3 import PPO

from src.env import Cellular6GEnv
from src.evaluate import HeuristicBaseline, MODEL_PATH, evaluate

N_STEPS: int = 10_000


def _sample_inputs(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate n observations and matching raw-traffic vectors from the env."""
    env = Cellular6GEnv()
    obs_buf = np.empty((n, env.N_SECTORS), dtype=np.float32)
    raw_buf = np.empty((n, env.N_SECTORS), dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    for i in range(n):
        obs_buf[i] = obs
        raw_buf[i] = env.traffic_demand
        obs, _r, _term, trunc, _info = env.step(env.action_space.sample())
        if trunc:
            obs, _ = env.reset()
    return obs_buf, raw_buf


def profile() -> dict:
    """Time N_STEPS single-sample predictions for the heuristic and the PPO agent."""
    model = PPO.load(MODEL_PATH)
    heuristic = HeuristicBaseline()
    obs_buf, raw_buf = _sample_inputs(N_STEPS)

    # Warm-up (JIT / cache / torch lazy init)
    for i in range(5):
        heuristic.predict(raw_buf[i])
        model.predict(obs_buf[i], deterministic=True)

    t0 = time.perf_counter()
    for i in range(N_STEPS):
        heuristic.predict(raw_buf[i])
    t1 = time.perf_counter()
    heur_total = t1 - t0

    t0 = time.perf_counter()
    for i in range(N_STEPS):
        model.predict(obs_buf[i], deterministic=True)
    t1 = time.perf_counter()
    ppo_total = t1 - t0

    results = evaluate(model)
    mean_saved = float(np.mean([m["energy_saved_pct"] for m in results.values()]))
    mean_drate = float(np.mean([m["agent_deficit_rate_pct"] for m in results.values()]))

    heur_ms = heur_total / N_STEPS * 1e3
    ppo_ms = ppo_total / N_STEPS * 1e3

    return {
        "n_steps": N_STEPS,
        "heur_total_s": heur_total,
        "ppo_total_s": ppo_total,
        "heur_ms_per_step": heur_ms,
        "ppo_ms_per_step": ppo_ms,
        "overhead_ms_per_step": ppo_ms - heur_ms,
        "overhead_ratio": ppo_total / heur_total if heur_total > 0 else float("inf"),
        "mean_energy_saved_pct": mean_saved,
        "mean_deficit_rate_pct": mean_drate,
    }


def print_report(p: dict) -> None:
    sep = "=" * 72
    print(sep)
    print(" COMPUTATIONAL COMPLEXITY PROFILER — Heuristic vs PPO Inference")
    print(sep)
    print(f" Inference samples          : {p['n_steps']:,}")
    print("-" * 72)
    print(f"{'':28s}{'Heuristic':>18s}{'PPO Agent':>18s}")
    print(f"{' Total inference time (s)':28s}{p['heur_total_s']:>18.4f}{p['ppo_total_s']:>18.4f}")
    print(f"{' Avg time / step (ms)':28s}{p['heur_ms_per_step']:>18.4f}{p['ppo_ms_per_step']:>18.4f}")
    print("-" * 72)
    print(f" Compute overhead / step    : {p['overhead_ms_per_step']:.4f} ms "
          f"({p['overhead_ratio']:.1f}x slower than heuristic)")
    print(f" Mean energy saved vs heur. : {p['mean_energy_saved_pct']:.2f} %")
    print(f" Mean capacity-deficit rate : {p['mean_deficit_rate_pct']:.2f} %")
    print(sep)
    # Trade-off conclusion
    day_steps = 96
    extra_ms_day = p["overhead_ms_per_step"] * day_steps
    print(" TRADE-OFF CONCLUSION")
    print(
        f"   The PPO controller adds {p['overhead_ms_per_step']:.3f} ms of inference "
        f"latency per control step (≈ {extra_ms_day:.1f} ms of extra compute per "
        f"{day_steps}-step day) while reducing base-station beam energy by "
        f"{p['mean_energy_saved_pct']:.2f} % on average, at a "
        f"{p['mean_deficit_rate_pct']:.2f} % capacity-deficit rate."
    )
    verdict = (
        "NET POSITIVE — sub-millisecond-scale compute overhead is negligible "
        "relative to the radio-energy savings."
        if p["mean_energy_saved_pct"] > 0 and p["ppo_ms_per_step"] < 10.0
        else "NET NEGATIVE — compute cost or energy outcome does not justify DRL."
    )
    print(f"   Verdict: {verdict}")
    print(sep)


def main() -> None:
    print_report(profile())


if __name__ == "__main__":
    main()
