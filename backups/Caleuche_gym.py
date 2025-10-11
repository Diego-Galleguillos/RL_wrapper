import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from WorldManager import WorldManager  # Keep your existing methods (pause, step, reset, etc.)
import time, math

class ROS2OdomNode(Node):
    """Helper node to continuously receive WAM-V odometry."""
    def __init__(self):
        super().__init__('wamv_odom_node')
        self.last_odom = None
        self.create_subscription(
            Odometry,
            '/wamv/sensors/position/ground_truth_odometry',
            self._odom_callback,
            10
        )

    def _odom_callback(self, msg: Odometry):
        self.last_odom = msg


class CaleucheGymEnv:
    def __init__(self, step_limit=400, step_length=0.1):
        # Initialize ROS 2
        rclpy.init()
        # WorldManager for pausing/stepping/resetting Gazebo
        self.wc = WorldManager()
        # ROS2 node to get odometry
        self.odom_node = ROS2OdomNode()

        self.done = False
        self.step_counter = 0
        self.step_limit = step_limit
        self.step_length = step_length  # Not used yet but could be linked to sim steps
        self.obs = None
        self.goal_pose = [
        -523.4477405710335,  # x
        174.86326544353705,  # y
        -0.10003555069266182, # z
        -0.0013315821659977893, # qx
        -1.8278577294843213e-05, # qy
        0.47348236876522648,     # qz
        0.8808022894062543       # qw
        ]

    def get_observation(self, timeout=2.0):
        """Returns the 13-element WAM-V state vector from odometry."""
        start = time.time()
        while self.odom_node.last_odom is None and (time.time() - start) < timeout:
            rclpy.spin_once(self.odom_node, timeout_sec=0.01)

        if self.odom_node.last_odom is None:
            raise RuntimeError("No odometry message received within timeout")

        odom = self.odom_node.last_odom
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
            odom.twist.twist.angular.z
        ]
        return self.obs
    

    def get_error(self):
        """Compute Euclidean distance in XY to goal"""
        if self.obs is None:
            return None
        dx = self.goal_pose[0] - self.obs[0]
        dy = self.goal_pose[1] - self.obs[1]
        return math.sqrt(dx**2 + dy**2)

    def get_reward(self):
        """Reward based on negative XY distance to goal"""
        error = self.get_error()
        if error is None:
            return 0.0
        # Exponential decay reward (closer = higher)
        return math.exp(-error)

    def check_done(self):
        """Done if within 1 meter of goal or step limit reached"""
        error = self.get_error()
        if error is None:
            return False
        return error < 1.0 or self.step_counter >= self.step_limit

    def check_done(self):
        #if distance less than 1 meter done
        pass

    def pass_action(self, action):
        # Placeholder: implement action application here
        # Example: send thruster commands via self.wc or other interfaces
        pass

    def step(self, action):
        # Apply action
        self.pass_action(action)
        # Advance simulation n steps (example: 1 step per step_length)
        self.wc.n_steps()
        # Get observation
        obs = self.get_observation()
        # Compute reward
        reward = self.get_reward()
        # Update done
        self.step_counter += 1
        self.done = self.step_counter >= self.step_limit
        return obs, reward, self.done

    def reset(self):
        # Reset the world
        self.wc.model_reset()
        self.step_counter = 0
        self.done = False
        # Wait a few steps to stabilize simulation
        self.wc.n_steps(10)
        self.wc.unpause()
        obs = self.get_observation()
        self.obs = obs
        self.wc.pause()
        # Return initial observation
        return obs

    def shutdown(self):
        # Shutdown ROS2 properly
        rclpy.shutdown()


if __name__ == "__main__":
    env = CaleucheGymEnv(step_limit=10)
    try:
        # Reset environment to get initial state
        obs = env.reset()
        print("Initial observation:", obs)

        # Take a dummy action (replace with real action later)
        dummy_action = None
        for i in range(1000):
            obs, reward, done = env.step(dummy_action)
            #print("Observation after one step:", obs)
            print("Reward:", reward)
            #print("Done:", done)

    finally:
        # Always shutdown ROS2
        env.shutdown()