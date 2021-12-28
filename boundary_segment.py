
import numpy as np
import open3d as o3d


def plot_(segmented_ground, segmented_boundary):
    """
    Visualize segmentation results using Open3D

    Parameters
    ----------
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray
    segmented_boundary:
        Segmented ground as N-by-3 numpy.ndarray
    """

    # ground element:
    pcd_boundary = o3d.geometry.PointCloud()
    pcd_boundary.points = o3d.utility.Vector3dVector(segmented_boundary)
    pcd_boundary.colors = o3d.utility.Vector3dVector(
        [
            [0.372]*3 for i in range(segmented_boundary.shape[0])
        ]
    )

    # # not ground boundary:
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)


    # visualize:
    o3d.visualization.draw_geometries([pcd_ground, pcd_boundary])


def ground_segmentation(data):
    """
    Segment ground plane from Velodyne measurement

    Parameters
    ----------
    data: numpy.ndarray
        Velodyne measurements as N-by-3 numpy.ndarray

    """
    # TODO  plane segmentation

    # 可视化坐标系
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])   #The x, y, z axis will be rendered as red, green, and blue arrows respectively.

    N, _ = data.shape

    #
    # pre-processing: filter by surface normals
    #
    # first, filter by surface normal
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(data)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=9))

    # 可视化原始点云
    # o3d.visualization.draw_geometries([pcd_original, axis])

    # keep points whose surface normal is approximate to z-axis for ground plane segementation:
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_y = np.abs(normals[:, 1])

    # 一般边坡15-30度
    idx_downsampled = (angular_distance_to_y > np.cos(np.pi / 6)) & (angular_distance_to_y < np.cos(np.pi / 12))
    downsampled = data[idx_downsampled]

    #可视化满足法向要求的点
    # abc = o3d.geometry.PointCloud()
    # abc.points = o3d.utility.Vector3dVector(downsampled)
    # o3d.visualization.draw_geometries([abc])

    # plane segmentation with RANSAC
    # 使用open3d自带的ransac
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(downsampled)
    model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    # post-processing: get ground output by distance to segemented plane
    distance_to_ground = np.abs(np.dot(data, np.asarray(model[:3])) + model[3])

    idx_ground = distance_to_ground <= 0.3
    idx_segmented = np.logical_not(idx_ground)

    segmented_cloud = data[idx_segmented]
    segmented_ground = data[idx_ground]
    segmented_boundary_index = np.argsort(segmented_ground[:,0])[::-1]
    segmented_boundary = segmented_ground[segmented_boundary_index][:50]
    segmented_ground = segmented_ground[segmented_boundary_index][50:]

    return segmented_cloud, segmented_ground, segmented_boundary


if __name__ == '__main__':
    # 加载点云数据
    pco=np.genfromtxt('11.csv', dtype=np.str, delimiter=',')
    k = [m.split() for m in pco]
    pc = np.array(k).astype(np.float32)[:,:3]

    segmented_cloud, segmented_ground, segmented_boundary = ground_segmentation(data=pc)

    # visualize :
    plot_(segmented_ground, segmented_boundary)
