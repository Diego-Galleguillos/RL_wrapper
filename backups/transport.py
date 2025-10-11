import gz.transport14 as transport
from gz.msgs11.world_control_pb2 import WorldControl
from gz.msgs11.boolean_pb2 import Boolean
import time

SERVICE = '/world/task0/control'
node = transport.Node()


def precise_sleep(target_duration):
    """High-resolution sleep using spin–sleep hybrid."""
    start = time.perf_counter()
    end = start + target_duration
    # Sleep most of the time
    while True:
        now = time.perf_counter()
        remaining = end - now
        if remaining <= 0:
            break
        elif remaining > 0.002:
            time.sleep(0.001)
        else:
            # Busy wait for final microseconds
            pass


def send_world_control(pause=False):
    req = WorldControl()
    req.pause = pause
    ok, resp = node.request(
        SERVICE,
        req,
        request_type=WorldControl,
        response_type=Boolean,
        timeout=1000
    )
    return ok


def step_simulation(step_time_s):
    """Advance simulation approximately step_time_s real seconds."""
    # Pause → unpause → sleep → pause sequence
    send_world_control(pause=True)
    ok1 = send_world_control(pause=False)
    precise_sleep(step_time_s)
    ok2 = send_world_control(pause=True)
    return ok1 and ok2


def n_steps_simulation(n, step_time_s):
    for i in range(n):
        success = step_simulation(step_time_s)
        if not success:
            print(f"Step {i} failed")


if __name__ == "__main__":
    n_steps_simulation(10, 0.05)
