#!/usr/bin/env python3
"""
test_caleuche_policy.py

Run a deterministic rollout of your trained SAC policy on the Caleuche env.
Assumes you saved:
 - model at logs_caleuche/sac_caleuche_final (SB3 .zip)
 - vecnormalize at logs_caleuche/vecnormalize.pkl (if used)

Usage:
    python3 test_caleuche_policy.py
"""

import os
import time
import numpy as np
import torch
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize as VN
from stable_baselines3.common.monitor import Monitor

# --- Edit these if needed ---
MODEL_PATH = "logs_caleuche/sac_caleuche_final"   # SB3 loads with or without .zip
VECNORM_PATH = "logs_caleuche/vecnormalize.pkl"
N_EPISODES = 5
RENDER = True  # set False if running headless or you don't have a renderer
DELAY_BETWEEN_STEPS = 0.0  # add small delay for visualization if needed (seconds)

# --- Create ROS2 node (same as training) ---
from caleuche_gym_env import CaleucheGymEnv, ROS2OdomNode

if not rclpy.ok():
    rclpy.init()
odom_node = ROS2OdomNode()

# --- env factory (single env wrapped in DummyVecEnv) ---
EPISODE_LENGTH = 150  # keep consistent with training if relevant

def make_env():
    def _init():
        env = CaleucheGymEnv(step_limit=EPISODE_LENGTH, odom_node=odom_node)
        env = Monitor(env)  # monitor for compatibility with some wrappers/logging
        return env
    return _init

# Build a single-env DummyVecEnv (keeps same API you used for saving VecNormalize)
eval_env = DummyVecEnv([make_env()])

# If VecNormalize stats exist, load them into the eval_env
if os.path.exists(VECNORM_PATH):
    try:
        eval_env = VN.load(VECNORM_PATH, eval_env)
        print(f"Loaded VecNormalize from {VECNORM_PATH}")
    except Exception as e:
        print("Failed to load VecNormalize:", e)
        print("Proceeding without VecNormalize (this may change observations).")

# --- device autodetect ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Load model ---
if not os.path.exists(MODEL_PATH) and not os.path.exists(MODEL_PATH + ".zip"):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}(.zip). Please check MODEL_PATH.")

print("Loading model...")
model = SAC.load(MODEL_PATH, env=eval_env, device=device)
print("Model loaded. model.device =", model.device)

# --- Helper: robust reset & step parsing for VecEnv / Gym / Gymnasium differences ---
def safe_reset(env):
    # VecEnv.reset() may return obs or (obs, info)
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs, info = result, None
    return obs, info

def safe_step(env, action):
    """
    Handles both older gym step (obs, rew, done, info)
    and gymnasium step (obs, rew, terminated, truncated, info),
    and VecEnv batched returns.
    """
    result = env.step(action)
    # result can be 4-tuple or 5-tuple (and batched arrays)
    if len(result) == 5:
        obs, rew, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
    else:
        obs, rew, done, info = result
    return obs, rew, done, info

# --- Run evaluation episodes ---
episode_rewards = []
try:
    for ep in range(N_EPISODES):
        obs, _ = safe_reset(eval_env)
        # For VecEnv the obs is typically batched (shape (1, ...))
        done = np.zeros((eval_env.num_envs,), dtype=bool)
        ep_reward = 0.0
        step_i = 0
        print(f"\nStarting episode {ep+1}/{N_EPISODES}")
        while not done[0]:
            # model.predict accepts batched obs for VecEnv
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = safe_step(eval_env, action)

            # rew may be batch-like; reduce to scalar for env 0
            rew_arr = np.asarray(rew).reshape(-1)
            ep_reward += float(rew_arr[0])

            step_i += 1
            if RENDER:
                # call env.render() - your env may implement its own rendering
                try:
                    eval_env.render()
                except Exception:
                    # some VecEnvs require calling .render on the wrapped env:
                    try:
                        eval_env.envs[0].render()
                    except Exception:
                        pass

            if DELAY_BETWEEN_STEPS > 0:
                time.sleep(DELAY_BETWEEN_STEPS)

            # safety guard to prevent infinite loops
            if step_i > (EPISODE_LENGTH + 1000):
                print("Episode exceeded max steps, breaking.")
                break

        print(f"Episode {ep+1} reward: {ep_reward:.3f}  steps: {step_i}")
        episode_rewards.append(ep_reward)

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C). Exiting evaluation loop.")

finally:
    # cleanup ROS and envs
    try:
        eval_env.close()
    except Exception:
        pass
    try:
        odom_node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

print("\nAll done. Eval rewards:", episode_rewards)
