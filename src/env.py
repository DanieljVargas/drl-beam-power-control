"""Sectorized beamforming energy optimization environment for a 6G cell."""
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Cellular6GEnv(gym.Env):
    """4-sector (N, S, E, W) beamforming power-control environment.

    Observation:
        Box(low=0.0, high=1.0, shape=(4,)) — normalized traffic demand per sector.

    Action:
        MultiDiscrete([3, 3, 3, 3]) — per-sector beam power level:
            0 = Off  (0 Mbps,   energy cost 0.0)
            1 = Low  (150 Mbps, energy cost 1.0)
            2 = High (300 Mbps, energy cost 2.5)

    Scenarios (selected at reset via ``options={"scenario": ...}``):
        - "standard":           diurnal sine per sector with phase offsets.
        - "stadium_event":      one sector receives sudden massive spikes.
        - "night_fluctuations": low baseline with sparse random bursts.
        - "mobility_shift":     a crowd drifts sector 0 -> 1 -> 2 over the day.
    """

    metadata = {"render_modes": []}

    N_SECTORS: int = 4
    SECTOR_NAMES = ("N", "S", "E", "W")
    SCENARIOS = ("standard", "stadium_event", "night_fluctuations", "mobility_shift")

    TRAFFIC_MIN: float = 0.0
    TRAFFIC_MAX: float = 300.0

    POWER_CAPACITY = np.array([0.0, 150.0, 300.0], dtype=np.float32)
    POWER_COST = np.array([0.0, 1.0, 2.5], dtype=np.float32)

    # Reward weights
    W_DEFICIT: float = 25.0  # penalty per (deficit / 150 Mbps) per sector
    W_OVER: float = 1.0      # penalty per (excess capacity / 150 Mbps) per sector

    def __init__(self, day_length: int = 96, noise_std: float = 12.0) -> None:
        super().__init__()
        self.day_length = day_length
        self.noise_std = noise_std

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N_SECTORS,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * self.N_SECTORS)

        self.time_step: int = 0
        self.scenario: str = "standard"
        self.traffic_demand: np.ndarray = np.zeros(self.N_SECTORS, dtype=np.float32)

        # Per-episode scenario parameters
        self._phase = np.zeros(self.N_SECTORS, dtype=np.float32)
        self._event_sector: int = 0
        self._event_start: int = 0
        self._event_end: int = 0
        self._mobility_drift: float = 0.0

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        return (self.traffic_demand / self.TRAFFIC_MAX).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Traffic generators
    # ------------------------------------------------------------------ #
    def _traffic_standard(self, t: int) -> np.ndarray:
        """Diurnal sine per sector, phase-offset, range ~[0, 280] + noise."""
        amp = 140.0
        base = amp * (1.0 + np.sin(2.0 * np.pi * t / self.day_length + self._phase))
        noise = self.np_random.normal(0.0, self.noise_std, size=self.N_SECTORS)
        return np.clip(base + noise, self.TRAFFIC_MIN, self.TRAFFIC_MAX)

    def _traffic_stadium(self, t: int) -> np.ndarray:
        """Low ambient traffic everywhere; one sector spikes during an event window."""
        ambient = 25.0 + self.np_random.normal(0.0, 8.0, size=self.N_SECTORS)
        traffic = np.clip(ambient, self.TRAFFIC_MIN, self.TRAFFIC_MAX)
        if self._event_start <= t < self._event_end:
            spike = 270.0 + self.np_random.normal(0.0, 15.0)
            traffic[self._event_sector] = float(
                np.clip(spike, self.TRAFFIC_MIN, self.TRAFFIC_MAX)
            )
        return traffic

    def _traffic_night(self, t: int) -> np.ndarray:
        """Mostly idle sectors with sparse, highly random bursts."""
        traffic = np.zeros(self.N_SECTORS, dtype=np.float64)
        for s in range(self.N_SECTORS):
            if self.np_random.random() < 0.20:
                burst = self.np_random.uniform(20.0, 260.0)
                traffic[s] = burst
            elif self.np_random.random() < 0.30:
                traffic[s] = self.np_random.uniform(1.0, 12.0)
        return np.clip(traffic, self.TRAFFIC_MIN, self.TRAFFIC_MAX)

    def _traffic_mobility(self, t: int) -> np.ndarray:
        """A crowd drifts directionally through sectors 0 -> 1 -> 2 over the day.

        The crowd centroid moves linearly in sector-index space (with a small
        Markovian random-walk perturbation stored in ``_mobility_drift``).
        Load is distributed to adjacent sectors via a triangular kernel so
        hand-offs are gradual. Sector 3 stays at ambient all day.
        """
        step = 2.0 / max(self.day_length - 1, 1)
        self._mobility_drift += self.np_random.normal(0.0, 0.04)
        self._mobility_drift = float(np.clip(self._mobility_drift, -0.4, 0.4))
        center = float(np.clip(step * t + self._mobility_drift, 0.0, 2.0))

        idx = np.arange(self.N_SECTORS, dtype=np.float64)
        weights = np.maximum(0.0, 1.0 - np.abs(idx - center))
        wsum = weights.sum()
        crowd = 240.0 * weights / (wsum if wsum > 0 else 1.0)

        ambient = 10.0
        noise = self.np_random.normal(0.0, 8.0, size=self.N_SECTORS)
        return np.clip(crowd + ambient + noise, self.TRAFFIC_MIN, self.TRAFFIC_MAX)

    def _gen_traffic(self, t: int) -> np.ndarray:
        if self.scenario == "standard":
            out = self._traffic_standard(t)
        elif self.scenario == "stadium_event":
            out = self._traffic_stadium(t)
        elif self.scenario == "mobility_shift":
            out = self._traffic_mobility(t)
        else:
            out = self._traffic_night(t)
        return out.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def _compute_reward(self, demand: np.ndarray, action: np.ndarray) -> float:
        capacity = self.POWER_CAPACITY[action]
        energy_cost = float(np.sum(self.POWER_COST[action]))

        deficit = np.maximum(demand - capacity, 0.0)
        excess = np.maximum(capacity - demand, 0.0)

        deficit_pen = self.W_DEFICIT * float(np.sum(deficit)) / 150.0
        over_pen = self.W_OVER * float(np.sum(excess)) / 150.0

        return -(energy_cost + deficit_pen + over_pen)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if options is not None and "scenario" in options:
            self.scenario = str(options["scenario"])
        else:
            self.scenario = self.SCENARIOS[self.np_random.integers(len(self.SCENARIOS))]
        if self.scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        self.time_step = 0
        self._phase = (
            np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32) * np.pi
            + self.np_random.uniform(-0.2, 0.2, size=self.N_SECTORS).astype(np.float32)
        )
        self._event_sector = int(self.np_random.integers(self.N_SECTORS))
        self._event_start = int(self.np_random.integers(20, 40))
        self._event_end = self._event_start + int(self.np_random.integers(20, 35))
        self._mobility_drift = 0.0

        self.traffic_demand = self._gen_traffic(self.time_step)
        return self._get_obs(), {"scenario": self.scenario}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64).reshape(self.N_SECTORS)

        reward = self._compute_reward(self.traffic_demand, action)

        self.time_step += 1
        self.traffic_demand = self._gen_traffic(self.time_step)

        terminated = False
        truncated = self.time_step >= self.day_length
        info: dict[str, Any] = {"scenario": self.scenario}

        return self._get_obs(), float(reward), terminated, truncated, info
