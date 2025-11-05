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
import math
import random


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
    def __init__(self, step_limit=400, odom_node=None):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
        self.wc = WorldManager()
        # use the provided node, or create one if not given
        self.odom_node = odom_node if odom_node is not None else ROS2OdomNode()
        self.odom_true = None

        self.done = False
        self.step_counter = 0
        self.step_limit = step_limit
        self.obs = None

        # Goals and start poses (copiados del ejemplo que diste)
        self.goal_pose_1 = [
            -522.1594574118119,
            179.87699221714647,
            -0.1630562136043632,
            0.001752661253542435,
            0.007011080777130351,
            0.654090869747751,
            0.7563814560375588
        ]

        self.goal_pose_2 = [
            -522.3291315795685,
            201.49660136429407,
            -0.07870714343902278,
            0.0020134109779915282,
            0.007179225016119804,
            0.6540433243041284,
            0.7564203426915529
        ]

        self.start_pose_1 = [
            -528.0274672293625,
            164.6023061425605,
            -0.04390476121577291,
            -0.001500540377608551,
            0.0021163920690534547,
            0.5666785635027967,
            0.8239348729903305
        ]

        self.start_pose_2 = [
            -517.822487364616,
            164.19673333821746,
            -0.05557483609303693,
            -0.0020505951064116,
            0.0008396348797597133,
            0.8268160950461664,
            0.5624679857961581
        ]

        self.start_pose_3 = [
            -521.9841173028476,
            163.0002783474852,
            -0.08069307057582592,
            -0.0027935716308597714,
            -0.001857133734800512,
            0.7865821051099876,
            0.6174765897850496
        ]

        self.goal_pose = self.goal_pose_1

        # Observación simple: error_x, error_y, twist_x, twist_y, yaw, yaw_twist
        obs_low = np.full(6, -np.inf, dtype=np.float32)
        obs_high = np.full(6, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: left/right thrusters [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

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

        # Extract linear velocity
        twist_x = odom.twist.twist.linear.x
        twist_y = odom.twist.twist.linear.y

        # Convert quaternion to yaw
        q = odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Angular velocity around z
        yaw_twist = odom.twist.twist.angular.z

        # Save a fuller odom_true for buoy checks
        self.odom_true = [
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

        # Simplified observation vector
        self.obs = np.array([
            error_x,
            error_y,
            twist_x,
            twist_y,
            yaw,
            yaw_twist
        ], dtype=np.float32)

        return self.obs

    def get_error(self):
        if self.obs is None:
            return 0.0, 0.0
        return float(self.obs[0]), float(self.obs[1])

    def get_reward(self):
        """
        Reward based on change in distance to the goal.
        - Positive reward if the robot gets closer.
        - Negative reward if the robot moves away.
        - Small step penalty to encourage faster completion.
        """
        error_x, error_y = self.get_error()
        current_dist = np.sqrt(error_x**2 + error_y**2)

        # Reward is the reduction in distance
        reward = self.prev_dist - current_dist

        # scale factor to increase magnitude (kept similar to tu versión)
        reward *= 20.0

        # Step penalty
        reward -= 0.1

        # Update previous distance for next step
        self.prev_dist = current_dist

        return float(reward)

    def check_done(self):
        """
        Returns: (terminated: bool, reward_extra: float)
         - reward_extra: additional reward/penalty when specific events happen
        """
        if self.odom_true is None:
            return False, 0.0

        error_x, error_y = self.get_error()
        dist = np.sqrt(error_x**2 + error_y**2)
        # If close to goal 1 -> switch to goal 2 and give small intermediate reward
        if dist < 4 and self.goal_pose == self.goal_pose_1:
            self.goal_pose = self.goal_pose_2
            print("Reached Goal 1, switching to Goal 2")
            # don't terminate the episode, give moderate reward to encourage reaching
            return False, 500.0

        # If close to goal 2 -> episode success (terminate) and give big reward
        if dist < 4 and self.goal_pose == self.goal_pose_2:
            self.goal_pose = self.goal_pose_1  # reset goal for next episodes
            return True, 1500.0

        # If step limit reached, reset goal and continue (handled by truncated in step)
        if self.step_counter >= self.step_limit:
            self.goal_pose = self.goal_pose_1

        # Check buoy collisions (assumes self.wc.buoys exists and maps names->(x,y,z))
        for buoy_name, (bx, by, bz) in getattr(self.wc, "buoys", {}).items():
            buoy_dist = np.sqrt((bx - self.odom_true[0])**2 + (by - self.odom_true[1])**2)
            if buoy_dist < 2.0:
                # collision -> terminate and negative large penalty
                self.goal_pose = self.goal_pose_1
                return True, -500.0

        if self.goal_pose == self.goal_pose_2:
            try:
                if self.odom_true[0] < -530.0 or self.odom_true[0] > -516.0:
                    # out of bounds in x when going to goal 2
                    self.goal_pose = self.goal_pose_1
                    print("Out of bounds X when heading to Goal 2")
                    return True, -100.0
                else:
                    print(self.odom_true[0])
            except Exception:
                print("Error checking out-of-bounds condition")
        # default: not done
        return dist < 3.5 or self.step_counter >= self.step_limit, 0.0

    def pass_action(self, action):
        # Update latest thrust values for publisher (agent can write faster than 10Hz)
        # Use same scaling as tu segundo ejemplo: actions in [0,1] -> thrust 0..100
        left = float(np.clip(action[0], 0.0, 1.0))
        right = float(np.clip(action[1], 0.0, 1.0))
        self.odom_node.left_thrust = 100.0 * left
        self.odom_node.right_thrust = 100.0 * right

    def step(self, action):
        self.pass_action(action)

        self.wc.n_steps()
        self.wc.unpause()
        obs = self.get_observation()
        self.obs = obs
        self.wc.pause()

        reward = self.get_reward()

        self.step_counter += 1

        terminated, reward_extra = self.check_done()            # task-specific done + extra reward
        truncated = self.step_counter >= self.step_limit  # timeout

        # Combine rewards: base + event extra - 1 (same idea que tu snippet)
        reward = reward + reward_extra - 1.0

        info = {}
        # optional debug print:
        # print(f"step={self.step_counter} reward={reward} term={terminated} trunc={truncated}")

        print(reward)

        return obs, float(reward), bool(terminated), bool(truncated), info

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
            random.seed(seed)

        # Reset step counter and done flag
        self.step_counter = 0
        self.done = False

        # Choose a random start pose among the provided ones
        random_pose = random.choice([
            self.start_pose_1,
            self.start_pose_2,
            self.start_pose_3
        ])

        # Reset world & buoys
        # note: model_reset signature assumed to accept the 7 pose args
        self.wc.model_reset(*random_pose)
        if hasattr(self.wc, "reset_buoys_simple"):
            try:
                self.wc.reset_buoys_simple()
            except Exception:
                pass

        # ensure simulator steps a bit to apply reset
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
