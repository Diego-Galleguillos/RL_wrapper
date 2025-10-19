# train_caleuche_sac.py
import os
import time
import signal
import numpy as np
import random

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import your environment
from caleuche_gym_env import CaleucheGymEnv, ROS2OdomNode

import rclpy

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize ROS2
if not rclpy.ok():
    rclpy.init()
odom_node = ROS2OdomNode()

# --- Configuration ---
LOGDIR = "logs_caleuche"
os.makedirs(LOGDIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(LOGDIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPISODE_LENGTH = 300
NUM_EPISODES = 750
TOTAL_TIMESTEPS = EPISODE_LENGTH * NUM_EPISODES

SEED = 42
EVAL_EPISODES = 5
CHECKPOINT_FREQ = 20_000  # timesteps

# --- Env factory ---
def make_env():
    def _init():
        env = CaleucheGymEnv(step_limit=EPISODE_LENGTH, odom_node=odom_node)
        env = Monitor(env)
        return env
    return _init

# --- Vectorized environment ---
vec_env = DummyVecEnv([make_env()])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="caleuche_chk"
)

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOGDIR, "best_model"),
    log_path=os.path.join(LOGDIR, "eval_logs"),
    eval_freq=10_000,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False
)

# --- SAC Model ---
model = SAC(
    policy="MultiInputPolicy",
    env=vec_env,
    verbose=1,
    seed=SEED,
    learning_rate=3e-4,
    buffer_size=33000,   # enough to learn, but not blow RAM
    batch_size=64,
    tau=0.01,
    gamma=0.99,
    train_freq=100,
    gradient_steps=100,
    learning_starts=1000,
    device=device,
)

# --- Graceful shutdown ---
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
            eval_env.close()
        except Exception:
            pass
        print("Exiting.")
        raise SystemExit()

signal.signal(signal.SIGINT, _sigint_handler)

# --- Train ---
print(f"Training SAC for {TOTAL_TIMESTEPS} timesteps ({NUM_EPISODES} episodes x {EPISODE_LENGTH} steps).")
start_time = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
training_time = time.time() - start_time
print(f"Training finished in {training_time:.1f} s")

# --- Save final model & VecNormalize ---
model_path = os.path.join(LOGDIR, "sac_caleuche_final")
model.save(model_path)
vecnorm_path = os.path.join(LOGDIR, "vecnormalize.pkl")
VecNormalize.save(vec_env, vecnorm_path)
print("Saved model to:", model_path)
print("Saved VecNormalize to:", vecnorm_path)
'''
# --- Deterministic evaluation ---
print("Running short deterministic evaluation...")
from stable_baselines3.common.vec_env import VecNormalize as VN

eval_env = DummyVecEnv([make_env()])
if os.path.exists(vecnorm_path):
    eval_env = VN.load(vecnorm_path, eval_env)
else:
    eval_env = DummyVecEnv([make_env()])

model = SAC.load(model_path, env=eval_env)

episode_rewards = []
for ep in range(5):
    obs, _ = eval_env.reset()
    done = [False]
    ep_reward = 0.0
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = eval_env.step(action)
        ep_reward += float(rew[0]) if isinstance(rew, (list, tuple, np.ndarray)) else float(rew)
        # safety to avoid infinite loops
        if ep_reward > 1e6:
            break
    episode_rewards.append(ep_reward)

print("Eval rewards:", episode_rewards)'''

# --- Cleanup ---
vec_env.close()
eval_env.close()
print("Training script finished.")
