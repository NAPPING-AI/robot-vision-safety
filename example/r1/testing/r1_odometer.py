import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

ChannelFactoryInitialize(0)

latest_state = None

def callback(message):
    global latest_state
    latest_state = message

subscriber = ChannelSubscriber(
    "rt/odommodestate",
    SportModeState_,
)
subscriber.Init(callback, 10)

try:
    while True:
        if latest_state is not None:
            print(
                "position =", list(latest_state.position),
                "velocity =", list(latest_state.velocity),
                "rpy =", list(latest_state.imu_state.rpy),
                "yaw_speed =", latest_state.yaw_speed,
            )

        time.sleep(1.0)  # Print once per second

except KeyboardInterrupt:
    pass