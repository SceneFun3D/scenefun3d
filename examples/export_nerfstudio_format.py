"""
Export the scenefun3d dataset to the nerfstudio format.

Before running this example, download the a few assets:
python -m data_downloader.data_asset_download --split test_set --download_only_one_video_sequence --download_dir data/test --dataset_assets hires_wide_intrinsics hires_poses hires_wide hires_depth

SceneFun3D Toolkit
"""

import os
import shutil
import numpy as np
import tyro
from typing import Annotated
from utils.data_parser import DataParser
import viser
import viser.transforms as tf

import json
import time
from pathlib import Path
from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from PIL import Image


def main(data_dir: Path = '/data/scenefun3d/data/',
         output_dir: str = '/data/opennerf/data/nerfstudio/scenefun3d/',
         csv_file: str = 'benchmark_file_lists/test_set_only_one_video.csv',
         max_dataset_size: int = 200):
  with open(csv_file, 'r') as f:
    lines = f.readlines()
  for line in lines[1:]:
    visit_id, video_id = line.strip().split(',')    
    export_scene(data_dir, output_dir, visit_id, video_id, max_dataset_size)


def export_scene(
  data_dir: Annotated[Path, tyro.conf.arg(help="Path to the dataset")],
  output_dir: Annotated[Path, tyro.conf.arg(help="Where to write the nerfstudio fileformats.")],
  visit_id: Annotated[str, tyro.conf.arg(help="Visit identifier")],
  video_id: Annotated[str, tyro.conf.arg(help="Video sequence identifier")],
  max_dataset_size: Annotated[int, tyro.conf.arg(help="Maximum number of images to export")],
  visualize: bool=False):
  
  verbose = True
  num_downscales = 2
  dataParser = DataParser(data_dir)
  
  # Copy laser_scan
  mesh_path = data_dir / visit_id / f'{visit_id}_laser_scan.ply'
  os.makedirs(output_dir / Path(visit_id), exist_ok=True)
  shutil.copy(mesh_path, output_dir / Path(visit_id))

  # Create output directories
  output_dir = output_dir / Path(visit_id) / Path(video_id)
  output_image_dir = output_dir / 'images'
  output_image_dir.mkdir(parents=True, exist_ok=True)
  output_depth_dir = output_dir / 'depths'
  output_depth_dir.mkdir(parents=True, exist_ok=True)
  
  # Check for the required data assets
  input_images = dataParser.get_rgb_frames(visit_id, video_id)
  input_depths = dataParser.get_depth_frames(visit_id, video_id)
  input_intrinsics = dataParser.get_camera_intrinsics(visit_id, video_id)
  input_extrinsics = dataParser.get_camera_trajectory(visit_id, video_id, pose_source="colmap")  # colmap=hires, arkit=lowres

  # Get timestamps
  input_images_timesteps = sorted(list(input_images.keys()))
  input_depths_timesteps = sorted(list(input_depths.keys()))
  intrinsics_timesteps = sorted(list(input_intrinsics.keys()))
  extrinsics_timesteps = sorted(list(input_extrinsics.keys()))

  valid_timesteps = []
  for timestep in input_images_timesteps:
    if timestep in input_depths_timesteps and \
       timestep in intrinsics_timesteps and \
       timestep in extrinsics_timesteps:
      valid_timesteps.append(timestep)
  
  num_images = len(valid_timesteps)
  idx = np.arange(num_images)
  if max_dataset_size != -1 and num_images > max_dataset_size:
      idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)
  
  valid_timesteps_filtered = [valid_timesteps[i] for i in idx]
  image_filenames = [Path(input_images[t]) for t in valid_timesteps_filtered]
  depth_filenames = [Path(input_depths[t]) for t in valid_timesteps_filtered]


  # Copy images to output directory
  copied_image_paths = process_data_utils.copy_images_list(
      image_filenames,
      image_dir=output_image_dir,
      verbose=verbose,
      num_downscales=num_downscales,
  )
  copied_depth_paths = process_data_utils.copy_images_list(
      depth_filenames,
      image_dir=output_depth_dir,
      verbose=verbose,
      num_downscales=num_downscales,
  )
  assert(len(copied_image_paths) == len(copied_depth_paths))
  copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]

  if visualize:
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
  
  out = {}
  out['camera_model'] = "OPENCV"
  out['orientation_override'] = "none"
  frames = []
  
  for i, t in enumerate(valid_timesteps_filtered):
    image_path = input_images[t]
    intrinsic_path = input_intrinsics[t]
    w, h, fx, fy, cx, cy = dataParser.read_camera_intrinsics(intrinsic_path)
    T = input_extrinsics[t]

    rot_x_180 = np.eye(3)
    a = np.pi
    rot_x_180[1, 1] = np.cos(a)
    rot_x_180[2, 2] = np.cos(a)
    rot_x_180[1, 2] = -np.sin(a)
    rot_x_180[2, 1] = np.sin(a)
    T_iphone2nerfstudio = np.eye(4)
    T_iphone2nerfstudio[0:3, 0:3] = rot_x_180

    R = T[:3, :3]
    t = T[:3, 3]

    frame = {}
    frame['fl_x'] = float(fx)
    frame['fl_y'] = float(fy)
    frame['cx'] = float(cx)
    frame['cy'] = float(cy)
    frame['w'] = float(w)
    frame['h'] = float(h)
    frame['file_path'] = f'images/frame_{i+1:05d}.jpg'
    frame['transform_matrix'] = (T @ T_iphone2nerfstudio).tolist()
    frames.append(frame)
    image = Image.open(image_path)
    if visualize:
      server.scene.add_camera_frustum(
        f"/frames/{i}/frustum",
        fov=2 * np.arctan2(w / 2, fx),
        aspect= w / h,
        scale=0.05,
        image=np.array(image)[::20, ::20],
        wxyz=tf.SO3.from_matrix(T[:3, :3]).wxyz,
        position=T[:3, 3] - input_extrinsics[valid_timesteps_filtered[0]][:3, 3])

  out['frames'] = frames
  with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=4)
  
  if visualize:
    while True:
      time.sleep(0.1)
  

if __name__ == "__main__":
  tyro.cli(main)