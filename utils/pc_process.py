import numpy as np
import open3d as o3d

# Combines multiple laser scans / point clouds
def combine_point_clouds(ply_filenames):
    pcd_combined = o3d.geometry.PointCloud()
    for filename in ply_filenames:

        pcd = o3d.io.read_point_cloud(filename)

        pcd_combined += pc_downsample(pcd)

    return pcd_combined
    
# Downsamples a point cloud using voxel downsampling
def pc_downsample(pcd_combined, voxel_size=0.005):
    downpcd = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

    return downpcd

# estimate normals of a point cloud
def pc_estimate_normals(pcd, radius = 0.1, max_nn = 16):
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = radius, max_nn = max_nn))

    return pcd
# def crop_extraneous_points_from_point_cloud(pcd, 
#                                             dbscan_eps = 0.02, 
#                                             dbscan_min_points = 10, 
#                                             return_bbox = False,
#                                             print_debug = False):
    
#     labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=print_debug))

#     max_label = labels.max()

#     if print_debug:
#         print(f"Point cloud has {max_label + 1} clusters")

#     unique_labels, label_counts = np.unique(labels, return_counts=True)
#     label_counts[unique_labels < 0] = 0

#     largest_cluster_label = unique_labels[np.argmax(label_counts)]
#     largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

#     largest_cluster_points = pcd.select_by_index(largest_cluster_indices)
    
#     # Calculate the bounding box of the largest cluster
#     bbox = largest_cluster_points.get_oriented_bounding_box()
    
#     if print_debug:
#         print(f"Initial point cloud: {pcd}")

#     pcd_cropped = pcd.crop(bbox)

#     if print_debug:
#         print(f"Point cloud after cropping: {pcd_cropped}")

#     if return_bbox:
#         bbox.color = (1, 0, 0) # change bbox color for better visualization
#         return pcd_cropped, bbox
#     else:
#         return pcd_cropped

def crop_extraneous_points_from_point_cloud(pcd, 
                                            dbscan_eps = 0.02, 
                                            dbscan_min_points = 10, 
                                            return_bbox = False,
                                            print_debug = False):
    
    labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=print_debug))

    max_label = labels.max()

    if print_debug:
        print(f"Point cloud has {max_label + 1} clusters")

    unique_labels, label_counts = np.unique(labels, return_counts=True)

    label_counts[unique_labels < 0] = 0

    largest_cluster_label = unique_labels[np.argmax(label_counts)]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # needs Open3D version > 0.9.0 
    # largest_cluster_points = pcd.select_by_index(largest_cluster_indices)
    
    # For Open3D version = 0.9.0
    pcd_points = np.array(pcd.points)
    pcd_points = pcd_points[largest_cluster_indices]
    largest_cluster_points = o3d.geometry.PointCloud()
    largest_cluster_points.points = o3d.utility.Vector3dVector(pcd_points)
    
    # Calculate the bounding box of the largest cluster
    bbox = largest_cluster_points.get_oriented_bounding_box()
    
    if print_debug:
        print(f"Initial point cloud: {pcd}")

    # needs Open3D version > 0.9.0 
    # pcd_cropped = pcd.crop(bbox)

    # For Open3D version = 0.9.0
    pcd_cropped = crop_point_cloud(pcd, bbox)

    if print_debug:
        print(f"Point cloud after cropping: {pcd_cropped}")

    if return_bbox:
        bbox.color = (1, 0, 0) # change bbox color for better visualization
        return pcd_cropped, bbox
    else:
        return pcd_cropped


# Crop function is malfunctioning in Open3d==0.9.0 - https://github.com/isl-org/Open3D/issues/3284
def crop_point_cloud(pcd, bbox):
    point_cloud_np = np.asarray(pcd.points)
    point_cloud_np_colors = np.asarray(pcd.colors)

    # mask = bbox.get_point_indices_within_bounding_box(pcd.points) # get_point_indices_within_bounding_box() does not work well for Open3d==0.9.0

    # Define a boolean mask to filter points within the bounding box
    mask = np.logical_and(np.all(point_cloud_np >= bbox.get_min_bound(), axis=1),
                        np.all(point_cloud_np <= bbox.get_max_bound(), axis=1))

    # Apply the mask to extract the cropped point cloud
    cropped_point_cloud = o3d.geometry.PointCloud()
    cropped_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np[mask])
    cropped_point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_np_colors[mask])

    return cropped_point_cloud