"""Exhaustive test suite for the sectorized-beamforming PPO agent.

Evaluates the trained model (models/ppo_6g_model.zip) on all scenarios and
asserts energy-saving, safety (deficit-rate) and inference-latency targets.
"""
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

os.makedirs("tests", exist_ok=True)

from src.env import Cellular6GEnv  # noqa: E402
from src.evaluate import MODEL_PATH, SCENARIOS, evaluate, print_report  # noqa: E402

MAX_DEFICIT_RATE_PCT: float = 5.0
MAX_INFERENCE_MS: float = 10.0
N_LATENCY_SAMPLES: int = 1000


def _measure_inference_ms(model: PPO) -> float:
    """Return the mean single-sample predict() latency in milliseconds."""
    env = Cellular6GEnv()
    obs, _ = env.reset(seed=0)
    samples = np.empty((N_LATENCY_SAMPLES, env.N_SECTORS), dtype=np.float32)
    for i in range(N_LATENCY_SAMPLES):
        samples[i] = obs
        obs, _r, _term, trunc, _info = env.step(env.action_space.sample())
        if trunc:
            obs, _ = env.reset()
    model.predict(samples[0], deterministic=True)  # warm-up
    t0 = time.perf_counter()
    for i in range(N_LATENCY_SAMPLES):
        model.predict(samples[i], deterministic=True)
    return (time.perf_counter() - t0) / N_LATENCY_SAMPLES * 1e3


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"FAIL: trained model not found at {MODEL_PATH}. Run `python -m src.train` first.")
        sys.exit(1)

    print(f"[test_agent] Loading model from {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    print(f"[test_agent] Evaluating on scenarios: {list(SCENARIOS)}")
    results = evaluate(model)
    print_report(results)

    print("[test_agent] Measuring inference latency...")
    infer_ms = _measure_inference_ms(model)
    print(f"  PPO avg inference / step: {infer_ms:.4f} ms")
    print("-" * 72)

    failures: list[str] = []

    covered = set(results.keys())
    expected = {"standard", "stadium_event", "night_fluctuations", "mobility_shift"}
    if not expected.issubset(covered):
        failures.append(f"missing scenarios in evaluation: {sorted(expected - covered)}")

    for sc, m in results.items():
        if not (m["energy_saved_pct"] > 0.0):
            failures.append(
                f"{sc}: energy_saved_pct={m['energy_saved_pct']:.2f}% (must be > 0.0%)"
            )
        if not (m["agent_deficit_rate_pct"] < MAX_DEFICIT_RATE_PCT):
            failures.append(
                f"{sc}: deficit_rate={m['agent_deficit_rate_pct']:.2f}% "
                f"(must be < {MAX_DEFICIT_RATE_PCT}%)"
            )

    if not (infer_ms < MAX_INFERENCE_MS):
        failures.append(
            f"inference latency {infer_ms:.3f} ms/step (must be < {MAX_INFERENCE_MS} ms)"
        )

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(
        f"PASS: all {len(results)} scenarios — energy_saved > 0%, "
        f"deficit_rate < {MAX_DEFICIT_RATE_PCT}%, inference < {MAX_INFERENCE_MS} ms/step."
    )


if __name__ == "__main__":
    main()
