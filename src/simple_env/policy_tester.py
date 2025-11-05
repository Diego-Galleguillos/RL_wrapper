# view_policy_caleuche.py
"""
Carga el modelo entrenado (SAC) y lo ejecuta determinísticamente sobre el entorno una cantidad
fija de episodios para que puedas observar su comportamiento en el simulador.

No guarda trayectorias ni plots — solo ejecuta "reset()" y "step()" para que el simulador muestre
la política actuando. Imprime la recompensa total por episodio.

Uso: python3 view_policy_caleuche.py
"""
import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from caleuche_gym_simple_env import CaleucheGymEnv, ROS2OdomNode
import rclpy

# Config
LOGDIR = "logs_caleuche"
MODEL_PATH = os.path.join(LOGDIR, "sac_caleuche_final.zip")
VECNORM_PATH = os.path.join(LOGDIR, "vecnormalize.pkl")
EPISODES = 10
EPISODE_LENGTH = 300  # máximo pasos por episodio

# init ROS2 and shared node
if not rclpy.ok():
    rclpy.init()
odom_node = ROS2OdomNode()

# env factory
def make_env():
    def _init():
        env = CaleucheGymEnv(step_limit=EPISODE_LENGTH, odom_node=odom_node)
        return Monitor(env)
    return _init

# prepare env
eval_env = DummyVecEnv([make_env()])

# load normalization if available
if os.path.exists(VECNORM_PATH):
    try:
        print("Loading VecNormalize stats from:", VECNORM_PATH)
        eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    except Exception as e:
        print("Failed loading VecNormalize, continuing without it:", e)

# load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Train first or change MODEL_PATH.")
print("Loading model from:", MODEL_PATH)
model = SAC.load(MODEL_PATH, env=eval_env)

print(f"Running policy for {EPISODES} episodes. Watch the simulator while this runs.")

for ep in range(EPISODES):
    obs = eval_env.reset()
    done = [False]
    ep_reward = 0.0
    steps = 0

    while not done[0] and steps < EPISODE_LENGTH:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = eval_env.step(action)
        ep_reward += float(rew[0]) if isinstance(rew, (list, tuple, np.ndarray)) else float(rew)
        steps += 1
        # small sleep so simulator has time to render between steps
        time.sleep(0.05)

    print(f"Episode {ep+1}/{EPISODES}: reward={ep_reward:.3f}, steps={steps}")

# cleanup
try:
    eval_env.close()
except Exception:
    pass

print("Done. If you had the simulator window open you should have seen the policy act.")
