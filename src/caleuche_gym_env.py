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
from sensor_msgs.msg import LaserScan, Image
import random

import cv2
from cv_bridge import CvBridge

class ROS2OdomNode(Node):
    """Node that spins in its own thread and publishes thrusters at 10Hz."""
    def __init__(self):
        super().__init__('wamv_odom_node')

        self.last_odom = None
        self.last_scan = None
        self.last_image = None
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
        
        #create subscription for ros2 topic /scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            qos
        )

        self.image_sub = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/oak_d_poe_camera_sensor/optical/image_raw',
            self.image_callback,
            qos
        )

        self.bridge = CvBridge()

        # publishers for thrusters (reliable)
        pub_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        self.right_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', pub_qos)
        self.left_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', pub_qos)

        self.right_thrust = 0.0
        self.left_thrust = 0.0


        self.frame_skip = 1      # process 1 frame every 2 messages
        self.frame_counter = 0
        self.image_width = 1280//6   # new smaller width
        self.image_height =  720//6  # new smaller height

        # timer publishes at 10Hz
        self.create_timer(0.1, self._publish_thrusters)

        # executor/thread so callbacks run even if main thread is busy
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

    def _scan_callback(self, msg):
        self.last_scan = msg


    def image_callback(self, msg: Image):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return  # skip this frame

        # Convert ROS Image to CV2
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # Ignore the top half (keep only the bottom half)
        height = cv_image.shape[0]
        cv_image = cv_image[int(height * 0.4):, :]


        # Resize to smaller resolution
        cv_image = cv2.resize(cv_image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)

        # Normalize to [0,1] float32
        self.last_image = cv_image.astype(np.float32) / 255.0


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
        self.odom_true = None

        self.done = False
        self.step_counter = 0
        self.step_limit = step_limit
        self.obs = None

        # Goal pose (x, y, z, qx, qy, qz, qw)
        '''        self.goal_pose = [
            -523.4477405710335,
            174.86326544353705,
            -0.10003555069266182,
            -0.0013315821659977893,
            -1.8278577294843213e-05,
            0.47348236876522648,
            0.8808022894062543
        ]'''
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

        # Starting position 2 (x, y, z, qx, qy, qz, qw)
        self.start_pose_2 = [
            -518.822487364616,
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
        #    'twist': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        #    'scan': spaces.Box(low=0.0, high=30.0, shape=(207,), dtype=np.float32),
        # Observation space: 13 odom values + 2 relative errors
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0, shape=(self.odom_node.image_height, self.odom_node.image_width, 3), dtype=np.float32)  # match resized HxW
        })



        # Action space: left/right thrusters [-100, 100]
        self.action_space = spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)

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

        scan = np.zeros(207, dtype=np.float32)
        scan = np.array(self.odom_node.last_scan.ranges, dtype=np.float32)
        scan[np.isinf(scan)] = self.odom_node.last_scan.range_max
        scan[np.isnan(scan)] = 0.0



        img = np.zeros((self.odom_node.image_height, self.odom_node.image_width, 3), dtype=np.float32)
        if self.odom_node.last_image is not None:
            img = self.odom_node.last_image

        twist_vec = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.angular.z
        ], dtype=np.float32)

        #    'twist': twist_vec,  # 3D: [vx, vy, yaw_rate]
        #    'scan': scan,        # 1D array of ranges

        obs = {
            'image': img         # H x W x 3
        }

        return obs

    def get_error(self):
        if self.odom_true is None:
            return 0.0, 0.0
        return self.odom_true[-2], self.odom_true[-1]
    
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

        # Optional: scale the reward
        reward *= 20.0  # scale factor to increase magnitude

        # Step penalty
        reward -= 0.1

        # Update previous distance for next step
        self.prev_dist = current_dist

        return reward


    def check_done(self):
        error_x, error_y = self.get_error()
        dist = np.sqrt(error_x**2 + error_y**2)
        print(dist, self.step_counter, self.step_limit)
        if dist < 3.5 and self.goal_pose == self.goal_pose_1:
            self.goal_pose = self.goal_pose_2
            print("Goal 1 reached, moving to Goal 2")
            dist = 100.0  # reset dist to avoid immediate done
            return False, 500
        elif dist < 3.5 and self.goal_pose == self.goal_pose_2:
            self.goal_pose = self.goal_pose_1
            return True, 1500
        if self.step_counter >= self.step_limit:
            self.goal_pose = self.goal_pose_1

        for buoy_name, (bx, by, bz) in self.wc.buoys.items():
            buoy_dist = np.sqrt((bx - self.odom_true[0])**2 + (by - self.odom_true[1])**2)
            if buoy_dist < 1.5:
                print(f"Collision with buoy {buoy_name} at distance {buoy_dist}")
                self.goal_pose = self.goal_pose_1
                return True, -500


        return dist < 3.5 or self.step_counter >= self.step_limit, 0

    def pass_action(self, action):
        # Update latest thrust values for publisher (agent can write faster than 10Hz)
        self.odom_node.left_thrust = 100.0 * float(np.clip(action[0], 0.0, 1.0))
        self.odom_node.right_thrust = 100.0 * float(np.clip(action[1], 0.0, 1.0))

    def step(self, action):
        self.pass_action(action)

        self.wc.n_steps()
        self.wc.unpause()
        obs = self.get_observation()
        self.obs = obs
        self.wc.pause()

        reward = self.get_reward()

        self.step_counter += 1

        terminated, reward_extra = self.check_done()            # task-specific done
        truncated = self.step_counter >= self.step_limit  # timeout
        reward += reward_extra - 1
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
        random_pose = random.choice([
            self.start_pose_1,
            self.start_pose_2,
            self.start_pose_3
        ])

        self.wc.model_reset(*random_pose)
        self.wc.reset_buoys_simple()
        # Reset the model using your existing WorldControl + Pose services
        #self.wc.model_reset()  # optionally pass custom x,y,z,qx,qy,qz,qw

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
