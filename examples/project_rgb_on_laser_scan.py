"""
Projects the RGB color of the iPad camera frames on the 3D points of the Faro laser scan.

This example script demonstrates how to utilize the data assets and helper scripts in the SceneFun3D dataset. 

Before running this example, download the required assets:
> python -m data_downloader.data_asset_download --split custom --download_dir data --visit_id 420673 --video_id 42445198 --dataset_assets hires_wide_intrinsics laser_scan_5mm hires_poses hires_wide hires_depth crop_mask

SceneFun3D Toolkit
"""

import numpy as np
import tyro
from typing import Annotated
from utils.data_parser import DataParser
from utils.fusion_util import PointCloudToImageMapper
from tqdm import tqdm
import viser
import time


def main(
  data_dir: Annotated[str, tyro.conf.arg(help="Path to the dataset")],
  visit_id: Annotated[str, tyro.conf.arg(help="Visit identifier")] = '420673',
  video_id: Annotated[str, tyro.conf.arg(help="Video sequence identifier")] = '42445198',
  coloring_asset: Annotated[int, tyro.conf.arg(help="RGB data asset to project color to laser scan")] = "hires_wide",
  crop_extraneous: Annotated[bool, tyro.conf.arg(help="Whether to crop extraneous points from laser scan")] = True,
  cut_bound: Annotated[float, tyro.conf.arg(help="Cut bound for the laser scan")] = 5,
  visibility_threshold: Annotated[float, tyro.conf.arg(help="Visibility threshold for the laser scan")] = 0.25
  ):

    print(f"Processing video_id {video_id} (visit_id: {visit_id}) ...")
    
    # Read all data assets
    dataParser = DataParser(data_dir)
    pointcloud = dataParser.get_laser_scan(visit_id)
    poses_from_traj = dataParser.get_camera_trajectory(visit_id, video_id, pose_source="colmap")
    rgb_frame_paths = dataParser.get_rgb_frames(visit_id, video_id, data_asset_identifier="hires_wide")
    depth_frame_paths = dataParser.get_depth_frames(visit_id, video_id, data_asset_identifier="hires_depth")
    intrinsics_paths = dataParser.get_camera_intrinsics(visit_id, video_id, data_asset_identifier="hires_wide_intrinsics")

    # Process pointcloud
    if crop_extraneous:
        pointcloud = dataParser.get_cropped_laser_scan(visit_id, pointcloud)
    point_positions = np.array(pointcloud.points)
    num_points = point_positions.shape[0]

    # Compute the 3D-2D mapping
    width, height, _, _, _, _ = dataParser.read_camera_intrinsics(next(iter(intrinsics_paths.values())))
    point2img_mapper = PointCloudToImageMapper((int(width), int(height)), visibility_threshold, cut_bound)

    # Iterate over the frames and compute the average color of the points
    counter = np.zeros((num_points, 1))
    sum_features = np.zeros((num_points, 3))
    skipped_frames = []
    sorted_timestamps = sorted(rgb_frame_paths.keys())
    subsample = 10  # for faster processing
    for cur_timestamp in tqdm(sorted_timestamps[::subsample], desc="Frames processing"):
        pose = dataParser.get_nearest_pose(cur_timestamp, poses_from_traj)
        if pose is None:
            print(f"Skipping frame {cur_timestamp}.")
            skipped_frames.append(cur_timestamp)
            continue

        color = dataParser.read_rgb_frame(rgb_frame_paths[cur_timestamp], normalize=True) 
        depth = dataParser.read_depth_frame(depth_frame_paths[cur_timestamp]) 
        intrinsic = dataParser.read_camera_intrinsics(intrinsics_paths[cur_timestamp], format ="matrix")
        
        # compute the 3D-2D mapping based on the depth
        mapping = np.ones([num_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, point_positions, depth, intrinsic)

        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue
        
        mask = mapping[:, 3]
        feat_2d_3d = color[mapping[:, 1], mapping[:,2], :]    
        valid_mask = mask != 0
        valid_indices = np.nonzero(valid_mask)[0]
        counter[valid_indices] += 1
        sum_features[valid_indices] += feat_2d_3d[valid_indices]

    if len(skipped_frames) > 0:
        print(f"{len(skipped_frames)} frames were skipped because of unmet conditions.")

    counter[counter==0] = 1e-5
    feat_bank = sum_features / counter
    feat_bank[feat_bank[:, 0:3] == [0.0, 0.0, 0.0]] = 169.0 / 255.0

    # Visualization
    server = viser.ViserServer()
    subsample = 10  # for faster visualization
    server.scene.add_point_cloud("pcd",
                                 points=point_positions[::subsample] - np.mean(point_positions, axis=0),
                                 colors=feat_bank[::subsample],
                                 point_size=0.01)
    while True:  # keep server alive
      time.sleep(0.2)

if __name__ == "__main__":
  tyro.cli(main)