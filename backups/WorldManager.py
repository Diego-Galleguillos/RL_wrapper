import gz.transport14 as transport
from gz.msgs11.world_control_pb2 import WorldControl
from gz.msgs11.boolean_pb2 import Boolean
import time
from gz.msgs11.pose_pb2 import Pose

class WorldManager:
    def __init__(self):
        # Change world name here:
        SERVICE = '/world/task0/control'
        self.node = transport.Node()
        self.service = SERVICE
        self.world_control = WorldControl()
    
    def pause(self):
        self.world_control.pause = True
        ok, resp = self.node.request(
            self.service,
            self.world_control,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        return ok
    
    def unpause(self):
        self.world_control.pause = False
        ok, resp = self.node.request(
            self.service,
            self.world_control,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        return ok
    
    def precise_sleep(self, target_duration):
        start = time.perf_counter()
        end = start + target_duration
        while time.perf_counter() < end:
            time.sleep(0.000005)  # Sleep in very short bursts
    
    def step(self, step_time_ms):
        self.unpause()
        self.precise_sleep(step_time_ms)
        self.pause()

    def n_steps(self, n, step_time_ms):
        for i in range(n):
            self.step(step_time_ms)
            self.precise_sleep(0.05)

    def shutdown(self, reset_time=True, reset_models=True):
        # Create a new WorldControl message (don’t reuse paused one)
        reset_msg = WorldControl()
        reset_msg.reset.all = False
        reset_msg.reset.time_only = reset_time and not reset_models
        reset_msg.reset.model_only = reset_models and not reset_time
        if reset_time and reset_models:
            reset_msg.reset.all = True
        
        ok, resp = self.node.request(
            self.service,
            reset_msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        print('Reset: ', ok)

    def reset(self):
        # Reset only model poses (not time, not full restart)
        reset_msg = WorldControl()
        reset_msg.reset.model_only = True
        
        ok, resp = self.node.request(
            self.service,
            reset_msg,
            request_type=WorldControl,
            response_type=Boolean,
            timeout=1000
        )
        print('Reset models:', ok)
        return ok
    
    def model_reset(self):
        """Teleports WAM-V to default position (-532, 165, 0)."""
        msg = Pose()
        msg.name = 'wamv'
        msg.position.x = -532.0
        msg.position.y = 165.0
        msg.position.z = 0.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0

        ok, _ = self.node.request(
            self.pose_service,
            msg,
            request_type=Pose,
            response_type=Boolean,
            timeout=2000
        )
        print(f"Model reset (teleport) ok={ok}")

        


    
if __name__ == "__main__":
    w = WorldManager()
    w.pause()
    w.n_steps(10, 0.1)
    w.shutdown()
    w.pause()

