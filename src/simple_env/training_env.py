# train_caleuche_sac.py
import os
import time
import signal
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import your environment (file name must match)
from caleuche_gym_simple_env import CaleucheGymEnv, ROS2OdomNode

import rclpy

# ROS2 node init (shared between envs)
if not rclpy.ok():
    rclpy.init()
odom_node = ROS2OdomNode()

# --- Configuration ---
LOGDIR = "logs_caleuche"
os.makedirs(LOGDIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(LOGDIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPISODE_LENGTH = 300        # steps per episode (requested)
NUM_EPISODES = 300         # episodes (requested)
TOTAL_TIMESTEPS = EPISODE_LENGTH * NUM_EPISODES  # 90_000

SEED = 42
CHECKPOINT_FREQ = 20_000    # save every N timesteps (adjustable)

# --- Env factory ---
def make_env(step_limit=EPISODE_LENGTH):
    def _init():
        # Create the env with step_limit equal to EPISODE_LENGTH and use shared odom_node
        env = CaleucheGymEnv(step_limit=step_limit, odom_node=odom_node)
        env = Monitor(env)
        return env
    return _init

# --- Create vectorized envs (single env) ---
vec_env = DummyVecEnv([make_env()])
# Normalize observations (recommended). Do not normalize reward for SAC typically.
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="caleuche_sac_chk"
)

# --- Build the model (SAC) ---
model = SAC(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    seed=SEED,
    buffer_size=100_000,
    learning_rate=3e-4,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    tensorboard_log=os.path.join(LOGDIR, "tb")
)

# Graceful shutdown handler (save on Ctrl+C)
def _sigint_handler(signum, frame):
    print("\nSIGINT received: saving model and VecNormalize stats...")
    try:
        timestamp = int(time.time())
        model.save(os.path.join(LOGDIR, f"sac_caleuche_interrupt_{timestamp}"))
        VecNormalize.save(vec_env, os.path.join(LOGDIR, f"vecnormalize_interrupt_{timestamp}.pkl"))
        print("Saved interrupt checkpoint.")
    except Exception as e:
        print("Error while saving on interrupt:", e)
    finally:
        try:
            vec_env.close()
        except Exception:
            pass
        print("Exiting.")
        raise SystemExit()

signal.signal(signal.SIGINT, _sigint_handler)

# --- Train ---
print(f"Training SAC for {TOTAL_TIMESTEPS} timesteps ({NUM_EPISODES} episodes x {EPISODE_LENGTH} steps).")
start_time = time.time()
# Only checkpoint_callback is used (no EvalCallback) to avoid normalization sync issues.
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
training_time = time.time() - start_time
print(f"Training finished in {training_time:.1f} s")

# --- Save final model & VecNormalize stats ---
model_path = os.path.join(LOGDIR, "sac_caleuche_final")
model.save(model_path)
vecnorm_path = os.path.join(LOGDIR, "vecnormalize.pkl")
VecNormalize.save(vec_env, vecnorm_path)
print("Saved model to:", model_path)
print("Saved VecNormalize to:", vecnorm_path)

# Cleanup
try:
    vec_env.close()
except Exception:
    pass

print("Training script finished.")
