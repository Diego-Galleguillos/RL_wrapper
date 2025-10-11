# play_caleuche.py
import os
import time
import argparse
import numpy as np

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import your environment (file name must match)
from caleuche_gym_env import CaleucheGymEnv, ROS2OdomNode

def make_env(step_limit, odom_node):
    def _init():
        env = CaleucheGymEnv(step_limit=step_limit, odom_node=odom_node)
        env = Monitor(env)  # keep same wrappers as training if possible
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Play a trained PPO policy in Caleuche sim")
    parser.add_argument("--model", default="logs_caleuche/ppo_caleuche_final.zip",
                        help="Path to the saved PPO model (.zip or folder)")
    parser.add_argument("--vec", default="logs_caleuche/vecnormalize.pkl",
                        help="Path to VecNormalize pickle saved during training (optional)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--step_limit", type=int, default=200, help="Env step limit per episode (same as training)")
    parser.add_argument("--render", action="store_true", help="Call env.render() each step if available")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between env steps (0.0 for as-fast-as-possible)")
    args = parser.parse_args()

    # Init ROS2 (only if not already initialized)
    if not rclpy.ok():
        rclpy.init()
    odom_node = ROS2OdomNode()

    # Create a DummyVecEnv with the same constructor signature used during training
    base_env = DummyVecEnv([lambda: CaleucheGymEnv(step_limit=args.step_limit, odom_node=odom_node)])

    # Load VecNormalize if available (keeps observation normalization consistent)
    env = base_env
    if os.path.exists(args.vec):
        try:
            env = VecNormalize.load(args.vec, base_env)
            # Use the VecNormalize in evaluation mode (no updating of running stats)
            env.training = False
            env.norm_reward = False
            print(f"Loaded VecNormalize from: {args.vec}")
        except Exception as e:
            print("Failed to load VecNormalize, continuing with unnormalized env. Error:", e)
            env = base_env
    else:
        print("VecNormalize file not found, using unnormalized env.")

    # Load model
    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model}")
    print("Loading model:", args.model)
    model = PPO.load(args.model, env=env)  # attaches env for convenience (not mandatory)

    # Play episodes
    episode_rewards = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = [False]  # VecEnv returns list-like done
        ep_reward = 0.0
        step = 0
        while not done[0]:
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, rew, done, info = env.step(action)
            # rew might be array-like (VecEnv) so coerce to float
            ep_reward += float(rew[0]) if isinstance(rew, (list, tuple, np.ndarray)) else float(rew)
            step += 1

            # optional rendering (if env implements render)
            if args.render:
                try:
                    # if wrapped in VecNormalize / DummyVecEnv, call .render() on the inner env
                    env.render()
                except Exception:
                    try:
                        base_env.render()
                    except Exception:
                        pass

            if args.sleep > 0:
                time.sleep(args.sleep)

            # safety guard to prevent infinite loops in buggy envs
            if step > args.step_limit + 50:
                print("Reached step guard; breaking episode.")
                break

        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{args.episodes} finished. Reward = {ep_reward:.3f} Steps = {step}")

        # small pause between episodes so ROS/Sim can stabilize
        time.sleep(0.5)

    print("All episodes finished. Rewards:", episode_rewards)

    # Cleanup
    try:
        env.close()
        base_env.close()
    except Exception:
        pass
    print("Playback script finished.")

if __name__ == "__main__":
    main()
