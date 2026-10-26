"""Train a PPO agent on the sectorized beamforming Cellular6GEnv."""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env import Cellular6GEnv

MODEL_DIR: str = "models"
MODEL_PATH: str = os.path.join(MODEL_DIR, "ppo_6g_model.zip")
TOTAL_TIMESTEPS: int = 500_000
N_ENVS: int = 4


def make_env() -> Cellular6GEnv:
    """Factory for a fresh env. Scenario is randomized on every reset."""
    return Cellular6GEnv()


def train(total_timesteps: int = TOTAL_TIMESTEPS, verbose: int = 1) -> PPO:
    """Train PPO (natively supports MultiDiscrete) and save the model."""
    env = DummyVecEnv([make_env for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        ent_coef=0.1,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.95,
        seed=0,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    return model


def main() -> None:
    train()
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
