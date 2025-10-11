import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from WorldManager import WorldManager
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from std_msgs.msg import Float64

import threading
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

class ROS2OdomNode(Node):
    """Node that spins in its own thread and publishes thrusters at 10Hz."""
    def __init__(self):
        super().__init__('wamv_odom_node')

        self.last_odom = None
        self._last_stamp = (0, 0)  # (sec, nanosec)
        self.prev_dist = 0

        # QoS tolerant for simulated topics
        qos = QoSProfile(depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry,
                                 '/wamv/sensors/position/ground_truth_odometry',
                                 self._odom_callback,
                                 qos)

        # publishers for thrusters (reliable)
        pub_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        self.right_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', pub_qos)
        self.left_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', pub_qos)

        self.right_thrust = 0.0
        self.left_thrust = 0.0

        # timer publishes at 10Hz
        self.create_timer(0.1, self._publish_thrusters)

        # executor/thread so callbacks run even if main thread is busy
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

    def _odom_callback(self, msg: Odometry):
        # store last message and stamp
        self.last_odom = msg
        self._last_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def _publish_thrusters(self):
        self.right_pub.publish(Float64(data=float(self.right_thrust)))
        self.left_pub.publish(Float64(data=float(self.left_thrust)))

    def destroy(self):
        # stop executor thread and destroy node when shutting down
        try:
            self._executor.remove_node(self)
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        if self._executor_thread.is_alive():
            self._executor_thread.join(timeout=1.0)
        try:
            super().destroy_node()
        except Exception:
            pass



class CaleucheGymEnv(gym.Env):
    """Gym environment for WAM-V using Gazebo & ROS2."""
class CaleucheGymEnv(gym.Env):
    def __init__(self, step_limit=400, odom_node=None):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
        self.wc = WorldManager()
        # use the provided node, or create one if not given
        self.odom_node = odom_node if odom_node is not None else ROS2OdomNode()

        self.done = False
        self.step_counter = 0
        self.step_limit = step_limit
        self.obs = None

        # Goal pose (x, y, z, qx, qy, qz, qw)
        self.goal_pose = [
            -523.4477405710335,
            174.86326544353705,
            -0.10003555069266182,
            -0.0013315821659977893,
            -1.8278577294843213e-05,
            0.47348236876522648,
            0.8808022894062543
        ]

        # Observation space: 13 odom values + 2 relative errors
        obs_low = np.full(15, -np.inf, dtype=np.float32)
        obs_high = np.full(15, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: left/right thrusters [-100, 100]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def get_observation(self, timeout=2.0):
        start = time.time()
        while self.odom_node.last_odom is None and (time.time() - start) < timeout:
            rclpy.spin_once(self.odom_node, timeout_sec=0.01)

        if self.odom_node.last_odom is None:
            raise RuntimeError("No odometry message received within timeout")

        odom = self.odom_node.last_odom

        # Compute relative error to goal in x and y
        error_x = self.goal_pose[0] - odom.pose.pose.position.x
        error_y = self.goal_pose[1] - odom.pose.pose.position.y

        self.obs = [
            # Position
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
            # Orientation
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
            # Linear velocity
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
            # Angular velocity
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
            # Relative errors
            error_x,
            error_y
        ]
        return self.obs

    def get_error(self):
        if self.obs is None:
            return 0.0, 0.0
        return self.obs[-2], self.obs[-1]

    def get_reward(self):
        """
        Reward function for WAM-V navigation task.

        - Dense reward: exponential decay with distance to goal.
        - Step penalty: small negative reward each step to encourage faster completion.
        """
        error_x, error_y = self.get_error()
        dist = np.sqrt(error_x**2 + error_y**2)

        # Dense reward: closer = higher
        reward = 10.0 * np.exp(-dist / 5.0)

        # Step penalty: small negative value to discourage lingering
        reward -= 0.1

        return reward


    def check_done(self):
        error_x, error_y = self.get_error()
        dist = np.sqrt(error_x**2 + error_y**2)
        print(dist, self.step_counter, self.step_limit)
        return dist < 0.5 or self.step_counter >= self.step_limit

    def pass_action(self, action):
        # Update latest thrust values for publisher (agent can write faster than 10Hz)
        self.odom_node.left_thrust = 100*float(np.clip(action[0], -1.0, 1.0))
        self.odom_node.right_thrust = 100*float(np.clip(action[1], -1.0, 1.0))

    def step(self, action):
        self.pass_action(action)

        self.wc.n_steps()
        self.wc.unpause()
        obs = self.get_observation()
        self.obs = obs
        self.wc.pause()

        reward = self.get_reward()

        self.step_counter += 1

        terminated = self.check_done()            # task-specific done
        truncated = self.step_counter >= self.step_limit  # timeout

        info = {}

        return obs, reward, terminated, truncated, info


    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and the WAM-V model in VRX.
        Compatible with Gymnasium + SB3.

        Returns:
            obs (np.array): Initial observation
            info (dict): Optional info dict (empty here)
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset step counter and done flag
        self.step_counter = 0
        self.done = False

        # Reset the model using your existing WorldControl + Pose services
        self.wc.model_reset()  # optionally pass custom x,y,z,qx,qy,qz,qw

        # Optional: update number of steps and unpause simulator briefly
        self.wc.n_steps()
        self.wc.unpause()

        # Get initial observation
        obs = self.get_observation()
        self.obs = obs

        # Store initial distance for delta-based reward
        error_x, error_y = self.get_error()
        self.prev_dist = np.sqrt(error_x**2 + error_y**2)

        # Pause simulator to keep control
        self.wc.pause()

        info = {}  # Gymnasium requires returning info dict
        return obs, info

    def shutdown(self):
        rclpy.shutdown()


if __name__ == "__main__":
    pass
