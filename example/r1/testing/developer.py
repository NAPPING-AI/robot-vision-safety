from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ChannelFactoryInitialize(0, "eth10") 

msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()

# Save the current controller before releasing it.
code, mode = msc.CheckMode()
if code != 0:
    raise RuntimeError(f"CheckMode failed: {code}")

previous_mode = mode["name"]
print("Current controller:", previous_mode)

# Enter low-level control
code, _ = msc.ReleaseMode()
print("Release return:", code)
print("Controller after release:", msc.CheckMode())

# Exit low-level control
# code, _ = msc.SelectMode("ai")
# print("Switch return:", code)
# print("Current mode:", msc.CheckMode())