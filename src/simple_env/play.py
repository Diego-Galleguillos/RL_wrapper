#!/usr/bin/env python3
"""
play.py

Play the best model deterministically, with correct rclpy init/shutdown handling.
"""
import os
import time
import argparse
import glob
import numpy as np

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# === Your environment ===
from caleuche_gym_simple_env import CaleucheGymEnv, ROS2OdomNode

# Defaults
LOG_DIR = "logs_caleuche"
BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model", "best_model.zip")
VEC_DEFAULT = os.path.join(LOG_DIR, "vecnormalize.pkl")

def find_best_or_latest_candidate(logdir=LOG_DIR):
    """Prefer best_model zip, else latest interrupt checkpoint, else latest zip in logdir."""
    best = os.path.join(logdir, "best_model", "best_model.zip")
    if os.path.exists(best):
        return best
    # newest interrupt
    intr = glob.glob(os.path.join(logdir, "ppo_caleuche_interrupt_*.zip"))
    if intr:
        return max(intr, key=os.path.getmtime)
    # fallback any zip
    anyz = glob.glob(os.path.join(logdir, "*.zip"))
    if anyz:
        return max(anyz, key=os.path.getmtime)
    # no candidate
    return None

def make_env_factory(step_limit, odom_node):
    """Return a callable to create the env that reuses odom_node created externally."""
    def _init():
        env = CaleucheGymEnv(step_limit=step_limit, odom_node=odom_node)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Play best PPO model for Caleuche (deterministic).")
    parser.add_argument("--model", default=None, help="Path to model .zip (overrides auto-detect)")
    parser.add_argument("--vec", default=None, help="Path to VecNormalize .pkl (optional)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--step_limit", type=int, default=180, help="Env step limit")
    parser.add_argument("--render", action="store_true", help="Call env.render() each step (if implemented)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between steps")
    args = parser.parse_args()

    # pick model if not provided
    model_path = args.model if args.model else find_best_or_latest_candidate()
    if model_path is None:
        raise FileNotFoundError(f"No model .zip found in '{LOG_DIR}'. Put best_model.zip in {LOG_DIR}/best_model or pass --model.")

    # pick vec if not provided: try exact vec file, else choose newest vecnormalize*.pkl
    if args.vec:
        vec_path = args.vec
    else:
        if os.path.exists(VEC_DEFAULT):
            vec_path = VEC_DEFAULT
        else:
            vec_candidates = glob.glob(os.path.join(LOG_DIR, "vecnormalize*.pkl"))
            vec_path = max(vec_candidates, key=os.path.getmtime) if vec_candidates else None

    print("Working dir:", os.getcwd())
    print("Model path:", model_path)
    print("VecNormalize path:", vec_path)

    # === Initialize rclpy once ===
    if not rclpy.ok():
        rclpy.init()
    # create and reuse a single odom node for the env(s)
    odom_node = ROS2OdomNode()

    # === Build vec env using the shared odom_node ===
    base_env = DummyVecEnv([make_env_factory(step_limit=args.step_limit, odom_node=odom_node)])

    # === Load VecNormalize if available ===
    env = base_env
    if vec_path and os.path.exists(vec_path):
        try:
            env = VecNormalize.load(vec_path, base_env)
            env.training = False
            env.norm_reward = False
            print("Loaded VecNormalize from:", vec_path)
        except Exception as e:
            print("Warning: failed to load VecNormalize. Continuing without it. Error:", e)
            env = base_env
    else:
        print("No VecNormalize loaded (not found). Observations will be unnormalized.")

    # === Load model (allow passing model without .zip extension) ===
    candidate_models = [model_path]
    if not model_path.endswith(".zip"):
        candidate_models.append(model_path + ".zip")
    # also try nested best_model folder if user passed logs_caleuche as model accidentally
    candidate_models.append(os.path.join(LOG_DIR, "best_model", "best_model.zip"))

    model = None
    for p in candidate_models:
        if p and os.path.exists(p):
            print("Loading model:", p)
            model = PPO.load(p, env=env)
            break

    if model is None:
        print("Model candidates checked:")
        for p in candidate_models:
            print("  ", p)
        raise FileNotFoundError("No valid model file found among candidates above.")

    # === Play episodes deterministically ===
    print("\nRunning deterministic policy (no exploration). Press Ctrl+C to stop.\n")
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = [False]
            ep_reward = 0.0
            step = 0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, done, info = env.step(action)
                # reward from VecEnv could be array-like
                ep_reward += float(rew[0]) if isinstance(rew, (list, tuple, np.ndarray)) else float(rew)
                step += 1

                if args.render:
                    try:
                        env.render()
                    except Exception:
                        try:
                            base_env.render()
                        except Exception:
                            pass

                if args.sleep and args.sleep > 0:
                    time.sleep(args.sleep)

                if step > args.step_limit + 200:
                    print("Step guard reached; breaking episode.")
                    break

            print(f"[Episode {ep+1}/{args.episodes}] reward={ep_reward:.3f} steps={step}")
            # small pause between episodes
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            env.close()
            base_env.close()
        except Exception:
            pass
        # destroy the node if it exists and shutdown rclpy
        try:
            odom_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Exited cleanly.")

if __name__ == "__main__":
    main()
