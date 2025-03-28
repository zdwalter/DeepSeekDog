import sys
import time
from typing import Any, Callable

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

TOPIC_RANGE_INFO = "rt/utlidar/range_info"

def pose_stamped_callback(pose_stamped_data: PoseStamped_):
    """Callback function to handle received PoseStamped data."""
    if pose_stamped_data is not None:
        # Extract the position (Point_) from the Pose_
        position = pose_stamped_data.pose.position
        print("Received PoseStamped data:")
        print(f"Position (x, y, z): ({position.x}, {position.y}, {position.z})")
        print("---")
    else:
        print("Received invalid or empty data.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    # Initialize the channel factory with the provided network interface
    ChannelFactoryInitialize(0, sys.argv[1])

    # Create a subscriber for the range info topic with a callback
    channel = ChannelSubscriber(TOPIC_RANGE_INFO, PoseStamped_)
    channel.Init(handler=pose_stamped_callback)

    try:
        # Keep the program running to allow callbacks to process data
        while True:
            time.sleep(0.1)  # Small delay to reduce CPU usage
    except KeyboardInterrupt:
        print("Exiting...")
        channel.Close()
