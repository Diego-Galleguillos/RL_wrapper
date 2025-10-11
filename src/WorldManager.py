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
   


if __name__ == "__main__":

    w = WorldManager()
    # Reset models and teleport WAM-V to default
    #w.model_reset()
    # Run 10 physics steps
    #w.n_steps(1000)
