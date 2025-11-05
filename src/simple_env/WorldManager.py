import gz.transport14 as transport
from gz.msgs11.world_control_pb2 import WorldControl
from gz.msgs11.boolean_pb2 import Boolean
from gz.msgs11.pose_pb2 import Pose


import time

class WorldManager:
    def __init__(self):
        # Services
        self.world_service = '/world/task0/control'
        self.pose_service = '/world/task0/set_pose'
        self.node = transport.Node()
        self.buoys = {
            "mb_marker_buoy_red_in":  (-528.0, 176.0, 0.0),
            "mb_marker_buoy_green_in":(-518.0, 176.0, 0.0),
            "mb_marker_buoy_red_out": (-528.0, 196.0, 0.0),
            "mb_marker_buoy_green_out":(-518.0, 196.0, 0.0),
        }



    def pause(self):
        msg = WorldControl()
        msg.pause = True
        ok, _ = self.node.request(
            self.world_service,
            msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )

        return ok

    def unpause(self):
        msg = WorldControl()
        msg.pause = False
        ok, _ = self.node.request(
            self.world_service,
            msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )

        return ok

    def step(self):
        """Advance simulation by exactly one physics step."""
        msg = WorldControl()
        msg.step = True
        ok, _ = self.node.request(
            self.world_service,
            msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        return ok

    def n_steps(self, n=25):
        """Run n discrete simulation steps."""
        self.pause()
        for _ in range(n):
            self.step()
            time.sleep(0.004)  # Small delay between steps

        

    def shutdown(self, reset_time=True, reset_models=True):
        msg = WorldControl()
        msg.reset.all = False
        msg.reset.time_only = reset_time and not reset_models
        msg.reset.model_only = reset_models and not reset_time
        if reset_time and reset_models:
            msg.reset.all = True
        ok, _ = self.node.request(
            self.world_service,
            msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        print('Reset:', ok)

    def model_reset(self, x=-532.0, y=165.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Reset model poses (WAM-V) to a given position and orientation."""
        # 1️⃣ Reset model poses in Gazebo
        msg = WorldControl()
        msg.reset.model_only = True
        ok, _ = self.node.request(
            self.world_service,
            msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        print('Model poses reset:', ok)

        # 2️⃣ Teleport WAM-V to specific coordinates
        pose = Pose()
        pose.name = 'wamv'
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        ok, _ = self.node.request(
            self.pose_service,
            pose,
            request_type=Pose,
            response_type=Boolean,
            timeout=1000
        )
        print(f'Teleport wamv to ({x}, {y}, {z}) ok={ok}')
        return ok
   

    def reset_buoys_simple(self, timeout=1000):
        print("Resetting buoys to original positions...")
        """
        Simple reset: clear model velocities and teleport buoys to original SDF poses.
        Mirrors the style of your existing model_reset for wamv.
        """
        # 1) Reset model poses (clear velocities)
        wc_msg = WorldControl()
        wc_msg.reset.model_only = True
        ok, _ = self.node.request(
            self.world_service,
            wc_msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=timeout
        )
        print('Model poses reset:', ok)

        # 2) Teleport buoys (original poses from your SDF)
        buoys = {
            "mb_marker_buoy_red_in":  (-528.0, 176.0, 0.0),
            "mb_marker_buoy_green_in":(-518.0, 176.0, 0.0),
            "mb_marker_buoy_red_out": (-528.0, 196.0, 0.0),
            "mb_marker_buoy_green_out":(-518.0, 196.0, 0.0),
        }

        all_ok = True
        for name, (x, y, z) in buoys.items():
            pose = Pose()
            pose.name = name
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(z)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            ok, _ = self.node.request(
                self.pose_service,
                pose,
                request_type=Pose,
                response_type=Boolean,
                timeout=timeout
            )
            print(f'Teleport {name} to ({x}, {y}, {z}) ok={ok}')
            all_ok = all_ok and bool(ok)

        return all_ok



if __name__ == "__main__":

    w = WorldManager()
    w.unpause()
    w.reset_buoys_simple()
    # Reset models and teleport WAM-V to default
    #w.model_reset()
    # Run 10 physics steps
    #w.n_steps(1000)
