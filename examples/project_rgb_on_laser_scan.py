"""
Projects the RGB color of the iPad camera frames on the 3D points of the Faro laser scan.

This example script demonstrates how to utilize the data assets and helper scripts in the SceneFun3D dataset. 

SceneFun3D Toolkit
"""

import numpy as np
import open3d as o3d
import os
import argparse
from utils.data_parser import DataParser
from utils.viz import viz_3d
from utils.fusion_util import PointCloudToImageMapper
from utils.pc_process import pc_estimate_normals
from tqdm import tqdm

##################
### PARAMETERS ###
##################
visibility_threshold = 0.25
cut_bound = 5
vis_result = True
##################
##################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data",
        help="Specify the path of the data"
    )

    parser.add_argument(
        "--visit_id",
        help="Specify the identifier of the scene"
    )

    parser.add_argument(
        "--video_id",
        help="Specify the identifier of the video sequence"
    )

    parser.add_argument(
        "--coloring_asset",
        default="hires_wide",
        choices=["hires_wide"], # TODO: add support for lowres_wide
        help="Specify the RGB data asset to use for projecting the color to the laser scan"
    )

    parser.add_argument(
        "--crop_extraneous",
        action="store_true",
        help="Specify whether to crop the extraneous points from the laser scan"
    )

    args = parser.parse_args()

    visit_id = str(args.visit_id)
    video_id = str(args.video_id)

    dataParser = DataParser(args.data_dir)

    print(f"Processing video_id {video_id} (visit_id: {visit_id}) ...")

    pcd = dataParser.get_laser_scan(visit_id)
    if args.crop_extraneous:
        pcd = dataParser.get_cropped_laser_scan(visit_id, pcd)

    locs_in = np.array(pcd.points)
    n_points = locs_in.shape[0]

    poses_from_traj = dataParser.get_camera_trajectory(visit_id, video_id, pose_source="colmap")

    rgb_frame_paths = dataParser.get_rgb_frames(
        visit_id, 
        video_id, 
        data_asset_identifier="hires_wide"
    )

    depth_frame_paths = dataParser.get_depth_frames(
        visit_id, 
        video_id, 
        data_asset_identifier="hires_depth"
    )

    intrinsics_paths = dataParser.get_camera_intrinsics(
        visit_id,
        video_id,
        data_asset_identifier="hires_wide_intrinsics"
    )

    w, h, _, _, _, _ = dataParser.read_camera_intrinsics(
        next(iter(intrinsics_paths.values()))
    )

    w = int(w)
    h = int(h)
    
    point2img_mapper = PointCloudToImageMapper(
        image_dim=(w, h),
        visibility_threshold=visibility_threshold,
        cut_bound=cut_bound
    )

    counter = np.zeros((n_points, 1))
    sum_features = np.zeros((n_points, 3))

    skipped_frames = []

    sorted_timestamps = sorted(rgb_frame_paths.keys())
    for cur_timestamp in tqdm(sorted_timestamps, desc="Frames processing"):

        pose = dataParser.get_nearest_pose(cur_timestamp, poses_from_traj)

        if pose is None:
            print(f"Skipping frame {cur_timestamp}.")
            skipped_frames.append(cur_timestamp)
            continue

        color = dataParser.read_rgb_frame(rgb_frame_paths[cur_timestamp], normalize=True) 
        depth = dataParser.read_depth_frame(depth_frame_paths[cur_timestamp]) 
        intrinsic = dataParser.read_camera_intrinsics(intrinsics_paths[cur_timestamp], format ="matrix")

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth, intrinsic)

        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mask = mapping[:, 3]
        feat_2d_3d = color[mapping[:, 1], mapping[:,2], :]

        counter[mask!=0] += 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    if len(skipped_frames) > 0:
        print(f"{len(skipped_frames)} frames were skipped because of unmet conditions.")

    counter[counter==0] = 1e-5
    feat_bank = sum_features / counter
    feat_bank[feat_bank[:, 0:3] == [0., 0., 0.]] = 169. / 255

    pcd.colors = o3d.utility.Vector3dVector(feat_bank)

    if vis_result:
        pcd = pc_estimate_normals(pcd)
        viz_3d([pcd], show_coordinate_system=False)

