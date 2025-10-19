# RL_wrapper
RL_wrapper for VRX Gazebo

add to bashrc:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

add to world:

<plugin name="gazebo_ros_api_plugin" filename="libgazebo_ros_api_plugin.so">
	  <robotNamespace>/</robotNamespace>
	</plugin>

	<plugin name="gz::sim::systems::WorldControl" filename="gz-sim-world-control-system">
	  <step_size>0.001</step_size>
	  <update_rate>1000</update_rate>
	</plugin>
	<!-- Allows pausing, stepping, resetting -->
	<plugin name="gz::sim::systems::WorldControl" filename="gz-sim-world-control-system"/>

	<!-- Allows setting model poses via /world/.../set_pose -->
	<plugin name="gz::sim::systems::PosePublisher" filename="gz-sim-pose-publisher-system"/>

	<!-- Optional: publish world stats -->
	<plugin name="gz::sim::systems::WorldStatsPublisher" filename="gz-sim-world-stats-publisher-system"/>

	<plugin name="gz::sim::systems::WorldControl" filename="gz-sim-world-control-system"/>


	<plugin
	  filename="libgz-sim-ros2-state-bridge-system.so"
	  name="gz::sim::systems::ROS2StateBridge">
	</plugin>

	<plugin
	  filename="libgz-sim-ros2-physics-system.so"
	  name="gz::sim::systems::ROS2Physics">
	</plugin>

how to use gz transport services:
gz service -s /world/task0/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --req "name: 'wamv', position: {x: -532.0, y: 165.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"