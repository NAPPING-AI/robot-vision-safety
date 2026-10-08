import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient

ChannelFactoryInitialize(0, sys.argv[1])

robot = LocoClient()
robot.SetTimeout(5.0)
robot.Init()

robot.StopMove()

ret = robot.SetFsmId(1)   # Damp()
print("Damp return:", ret)