# train_caleuche.py
import os
import time
import signal
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import your environment (file name must match)
from caleuche_gym_simple_env import CaleucheGymEnv, ROS2OdomNode

import rclpy
#ROS2 node

if not rclpy.ok():
    rclpy.init()

odom_node = ROS2OdomNode()

# --- Configuration ---
LOGDIR = "logs_caleuche"
os.makedirs(LOGDIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(LOGDIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPISODE_LENGTH = 500         # steps per episode (you requested)
NUM_EPISODES = 1000          # episodes
TOTAL_TIMESTEPS = EPISODE_LENGTH * NUM_EPISODES  # 80_000

SEED = 42
EVAL_EPISODES = 5
CHECKPOINT_FREQ = 20_000    # save every N timesteps (adjust as desired)

# --- Env factory ---
def make_env():
    def _init():
        # Create the env with step_limit equal to EPISODE_LENGTH
        env = CaleucheGymEnv(step_limit=EPISODE_LENGTH)
        # Wrap with Monitor so SB3 can read episode returns / lengths
        env = Monitor(env)
        return env
    return _init

# --- Create a single-vector env (DummyVecEnv) using the shared node ---
vec_env = DummyVecEnv([lambda: CaleucheGymEnv(step_limit=EPISODE_LENGTH, odom_node=odom_node)])

# Normalize observations (recommended). Don't normalize rewards here.
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="caleuche_chk"
)

# Evaluation callback (optional) - uses the same shared node
eval_env = DummyVecEnv([lambda: CaleucheGymEnv(step_limit=EPISODE_LENGTH, odom_node=odom_node)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)  # separate normalization for eval
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOGDIR, "best_model"),
    log_path=os.path.join(LOGDIR, "eval_logs"),
    eval_freq=10_000,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False
)


# --- Build the model (PPO) ---
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    seed=SEED,
    n_steps=1024,        # rollout length
    batch_size=64,       # mini-batch for policy update
    learning_rate=3e-4,  
    ent_coef=0.01,       # small entropy bonus to encourage exploration
    vf_coef=0.5,
    gamma=0.99,
    gae_lambda=0.95,     # GAE smoothing
    clip_range=0.2,
    tensorboard_log=os.path.join(LOGDIR, "tb")
)


# Graceful shutdown handler (save on Ctrl+C)
def _sigint_handler(signum, frame):
    print("\nSIGINT received: saving model and VecNormalize stats...")
    try:
        timestamp = int(time.time())
        model.save(os.path.join(LOGDIR, f"ppo_caleuche_interrupt_{timestamp}"))
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
print(f"Training for {TOTAL_TIMESTEPS} timesteps ({NUM_EPISODES} episodes x {EPISODE_LENGTH} steps).")
start_time = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
training_time = time.time() - start_time
print(f"Training finished in {training_time:.1f} s")

# --- Save final model & VecNormalize stats ---
model_path = os.path.join(LOGDIR, "ppo_caleuche_final")
model.save(model_path)
vecnorm_path = os.path.join(LOGDIR, "vecnormalize.pkl")
VecNormalize.save(vec_env, vecnorm_path)
print("Saved model to:", model_path)
print("Saved VecNormalize to:", vecnorm_path)

# --- Short deterministic evaluation ---
print("Running short deterministic evaluation...")
# Reload VecNormalize for evaluation to ensure consistent normalization
from stable_baselines3.common.vec_env import VecNormalize as VN
eval_env = DummyVecEnv([make_env()])
if os.path.exists(vecnorm_path):
    eval_env = VN.load(vecnorm_path, eval_env)
else:
    eval_env = DummyVecEnv([make_env()])

model = PPO.load(model_path, env=eval_env)
episode_rewards = []
for ep in range(5):
    obs = eval_env.reset()
    done = [False]
    ep_reward = 0.0
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = eval_env.step(action)
        # rew may be array-like because of VecEnv
        ep_reward += float(rew[0]) if isinstance(rew, (list, tuple, np.ndarray)) else float(rew)
        # safety to avoid infinite loops in case env buggy
        if ep_reward > 1e6:
            break
    episode_rewards.append(ep_reward)
print("Eval rewards:", episode_rewards)

# Cleanup
vec_env.close()
eval_env.close()
print("Training script finished.")
