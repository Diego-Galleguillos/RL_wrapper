import gz.transport14 as transport
from gz.msgs11.world_control_pb2 import WorldControl
from gz.msgs11.boolean_pb2 import Boolean

# Change world name here:
SERVICE = '/world/task0/control'

node = transport.Node()

# --- PAUSE ---
pause_req = WorldControl()
pause_req.pause = True

ok, resp = node.request(
    SERVICE,
    pause_req,
    request_type=WorldControl,
    response_type=Boolean,
    timeout=1000
)
print("Paused:", ok)

input("Press Enter to continue...")

# --- UNPAUSE ---
unpause_req = WorldControl()
unpause_req.pause = False

ok, resp = node.request(
    SERVICE,
    unpause_req,
    request_type=WorldControl,
    response_type=Boolean,
    timeout=1000
)
print("Unpaused:", ok)
