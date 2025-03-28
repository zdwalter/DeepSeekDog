from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
import cyclonedds.idl.types as types

def create_point_cloud_example():
    # 创建一个Header对象（这里简化为假设Header类已定义）
    # 由于代码片段中未给出Header类的完整定义，我们简单假设一个空的Header
    class Header:
        pass
    header = Header()

    # 创建PointField对象
    field1 = PointField_(
        name="x",
        offset=0,
        datatype=types.uint8(1),  # 假设为UINT8类型
        count=types.uint32(1)
    )
    field2 = PointField_(
        name="y",
        offset=1,
        datatype=types.uint8(1),
        count=types.uint32(1)
    )
    fields = [field1, field2]

    # 创建PointCloud2对象
    point_cloud = PointCloud2_(
        header=header,
        height=types.uint32(10),
        width=types.uint32(10),
        fields=fields,
        is_bigendian=False,
        point_step=types.uint32(2),
        row_step=types.uint32(20),
        data=[types.uint8(i) for i in range(200)],
        is_dense=True
    )

    # 打印点云信息
    print("PointCloud2信息:")
    print(f"Height: {point_cloud.height}")
    print(f"Width: {point_cloud.width}")
    print(f"Fields: {[field.name for field in point_cloud.fields]}")
    print(f"Is Big Endian: {point_cloud.is_bigendian}")
    print(f"Point Step: {point_cloud.point_step}")
    print(f"Row Step: {point_cloud.row_step}")
    print(f"Data Length: {len(point_cloud.data)}")
    print(f"Is Dense: {point_cloud.is_dense}")

    return point_cloud

if __name__ == "__main__":
    create_point_cloud_example()
