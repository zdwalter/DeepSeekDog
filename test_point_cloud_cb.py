import sys
import time
from typing import Any, Callable

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_

TOPIC_CLOUD = "rt/utlidar/cloud_deskewed"

def point_cloud_callback(cloud_data: PointCloud2_):
    """Callback function to handle received PointCloud2 data."""
    if cloud_data is not None:
        print("Received PointCloud2 data:")
        print(f"Height: {cloud_data.height}, Width: {cloud_data.width}")
        print(f"Point Step: {cloud_data.point_step}, Row Step: {cloud_data.row_step}")
        print(f"Is Dense: {cloud_data.is_dense}")
        print(f"Data Length: {len(cloud_data.data)} bytes")
        print("---")
    else:
        print("Received invalid or empty data.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    # Initialize the channel factory with the provided network interface
    ChannelFactoryInitialize(0, sys.argv[1])

    # Create a subscriber for the point cloud topic with a callback
    channel = ChannelSubscriber(TOPIC_CLOUD, PointCloud2_)
    channel.Init(handler=point_cloud_callback)

    try:
        # Keep the program running to allow callbacks to process data
        while True:
            time.sleep(0.1)  # Small delay to reduce CPU usage
    except KeyboardInterrupt:
        print("Exiting...")
        channel.Close()
