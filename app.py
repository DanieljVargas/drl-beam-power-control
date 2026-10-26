"""Streamlit dashboard for the sectorized-beamforming DRL proof of concept."""
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO

from src.env import Cellular6GEnv
from src.evaluate import (
    HeuristicBaseline,
    MODEL_PATH,
    POWER_CAPACITY,
    POWER_COST,
    SCENARIOS,
    SEED,
    STEPS,
    record_traffic_trace,
)

# Display sectors in natural compass order (clockwise) regardless of env index order.
_ENV_IDX = {name: i for i, name in enumerate(Cellular6GEnv.SECTOR_NAMES)}
COMPASS_ORDER = ("N", "E", "S", "W")
COMPASS_LABELS = ("North", "East", "South", "West")
COMPASS_IDX = [_ENV_IDX[c] for c in COMPASS_ORDER]

st.set_page_config(
    page_title="DRL Beamforming Energy Optimization",
    layout="wide",
)

st.title("DRL Beamforming Energy Optimization")


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(path: str) -> PPO:
    """Load and cache the trained PPO model."""
    return PPO.load(path)


def rollout_agent(
    env: Cellular6GEnv, model: PPO, scenario: str, steps: int, seed: int
) -> np.ndarray:
    """Roll out the PPO agent and return the chosen power level per (step, sector)."""
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    actions = np.empty((steps, env.N_SECTORS), dtype=np.int64)
    for t in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        actions[t] = np.asarray(a, dtype=np.int64).reshape(env.N_SECTORS)
        obs, _r, _term, _trunc, _info = env.step(a)
    return actions


def measure_inference_ms(model: PPO, n: int = 200) -> float:
    """Measure mean single-sample predict() latency in milliseconds."""
    rng = np.random.default_rng(0)
    samples = rng.random((n, Cellular6GEnv.N_SECTORS)).astype(np.float32)
    model.predict(samples[0], deterministic=True)  # warm-up
    t0 = time.perf_counter()
    for i in range(n):
        model.predict(samples[i], deterministic=True)
    return (time.perf_counter() - t0) / n * 1e3


def run_simulation(scenario: str) -> dict:
    """Run heuristic + PPO on the same seeded traffic trace and return all arrays/metrics."""
    env = Cellular6GEnv()
    model = load_model(MODEL_PATH)
    heuristic = HeuristicBaseline()

    traffic = record_traffic_trace(env, scenario, STEPS, SEED)  # (T, 4) Mbps

    base_actions = np.stack([heuristic.predict(traffic[t]) for t in range(STEPS)])
    agent_actions = rollout_agent(env, model, scenario, STEPS, SEED)

    base_cap = POWER_CAPACITY[base_actions]
    agent_cap = POWER_CAPACITY[agent_actions]

    base_energy = float(np.sum(POWER_COST[base_actions]))
    agent_energy = float(np.sum(POWER_COST[agent_actions]))
    base_deficit = float(np.sum(np.maximum(traffic - base_cap, 0.0)))
    agent_deficit = float(np.sum(np.maximum(traffic - agent_cap, 0.0)))
    saved_pct = (
        (base_energy - agent_energy) / base_energy * 100.0 if base_energy > 0 else 0.0
    )

    infer_ms = measure_inference_ms(model)

    return {
        "scenario": scenario,
        "traffic": traffic,
        "agent_cap": agent_cap,
        "base_cap": base_cap,
        "base_energy": base_energy,
        "agent_energy": agent_energy,
        "base_deficit": base_deficit,
        "agent_deficit": agent_deficit,
        "energy_saved_pct": saved_pct,
        "infer_ms": infer_ms,
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Beamforming PoC")
    st.markdown(
        "This proof of concept controls **per-sector beam power** (Off / Low / High) "
        "in a 4-sector 6G cell. A **PPO agent** learns to match provisioned "
        "capacity to spatial traffic demand, saving radio energy versus a "
        "traditional threshold **heuristic** while keeping capacity deficits low."
    )
    st.markdown(
        f"- Sectors: **{', '.join(COMPASS_LABELS)}**\n"
        f"- Power levels: Off = 0 Mbps · Low = 150 Mbps · High = 300 Mbps\n"
        f"- Horizon: **{STEPS}** steps · Seed: **{SEED}**"
    )
    st.divider()
    scenario = st.selectbox("Scenario", options=list(SCENARIOS), index=0)
    run = st.button("Run Simulation", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Execution (results cached in session_state so the slider doesn't re-simulate)
# --------------------------------------------------------------------------- #
if run:
    with st.spinner(f"Running heuristic + PPO on '{scenario}'..."):
        st.session_state["sim"] = run_simulation(scenario)

sim = st.session_state.get("sim")

if sim is None:
    st.info("Select a scenario and press **Run Simulation** in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Top metrics
# --------------------------------------------------------------------------- #
m1, m2, m3 = st.columns(3)
with m1:
    st.metric(
        label="Energy Saved vs Heuristic",
        value=f"{sim['energy_saved_pct']:.2f} %",
        delta=f"{sim['base_energy'] - sim['agent_energy']:+.1f} cost units",
    )
with m2:
    st.metric(
        label="Avg PPO Inference / Step",
        value=f"{sim['infer_ms']:.2f} ms",
    )
with m3:
    st.metric(
        label="Total Capacity Deficit (Agent)",
        value=f"{sim['agent_deficit']:.1f} Mbps",
        delta=f"{sim['agent_deficit'] - sim['base_deficit']:+.1f} vs heuristic",
        delta_color="inverse",
    )

st.caption(f"Scenario: **{sim['scenario']}**")

# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

# ---- Left: aggregate timeline -------------------------------------------- #
with left:
    x = np.arange(STEPS)
    agg_traffic = sim["traffic"].sum(axis=1)
    agg_agent_cap = sim["agent_cap"].sum(axis=1)

    fig_tl = go.Figure()
    fig_tl.add_trace(
        go.Scatter(
            x=x,
            y=agg_agent_cap,
            name="DRL Agent Capacity (aggregate)",
            mode="lines",
            line=dict(shape="hv", width=0, color="rgba(0,180,120,1.0)"),
            fill="tozeroy",
            fillcolor="rgba(0,180,120,0.25)",
        )
    )
    fig_tl.add_trace(
        go.Scatter(
            x=x,
            y=agg_traffic,
            name="Traffic Demand (aggregate)",
            mode="lines",
            line=dict(color="#d62728", width=2),
        )
    )
    fig_tl.update_layout(
        title="Cell-wide Traffic vs DRL Provisioned Capacity",
        xaxis_title="Time step",
        yaxis_title="Mbps (sum over 4 sectors)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

# ---- Right: spatial radar at selected time step -------------------------- #
with right:
    theta = list(COMPASS_LABELS) + [COMPASS_LABELS[0]]  # close the polygon

    # Use a placeholder so the chart renders ABOVE the slider but reacts to it.
    radar_slot = st.empty()
    t_sel = st.slider("Time Step", min_value=0, max_value=STEPS - 1, value=0)

    traffic_t = sim["traffic"][t_sel, COMPASS_IDX]
    agent_cap_t = sim["agent_cap"][t_sel, COMPASS_IDX]

    r_traffic = np.append(traffic_t, traffic_t[0])
    r_cap = np.append(agent_cap_t, agent_cap_t[0])

    fig_rd = go.Figure()
    fig_rd.add_trace(
        go.Scatterpolar(
            r=r_cap,
            theta=theta,
            name="Agent Capacity",
            fill="toself",
            line=dict(color="rgba(0,180,120,1.0)"),
            fillcolor="rgba(0,180,120,0.25)",
        )
    )
    fig_rd.add_trace(
        go.Scatterpolar(
            r=r_traffic,
            theta=theta,
            name="Traffic Demand",
            fill="toself",
            line=dict(color="#d62728"),
            fillcolor="rgba(214,39,40,0.20)",
        )
    )
    fig_rd.update_layout(
        title=f"Sector Beamforming — step {t_sel}",
        polar=dict(
            radialaxis=dict(range=[0, Cellular6GEnv.TRAFFIC_MAX], showticklabels=True),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    radar_slot.plotly_chart(fig_rd, use_container_width=True)
