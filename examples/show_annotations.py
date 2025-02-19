"""
Visualize annotations from the SceneFun3D dataset.

Before running this example, download the a few assets:
python -m data_downloader.data_asset_download --split custom --download_dir data --visit_id 420673 --video_id 42445198 --dataset_assets hires_wide_intrinsics laser_scan_5mm hires_poses hires_wide hires_depth crop_mask annotations descriptions motions arkit_mesh transform
python -m data_downloader.data_asset_download --split custom --download_dir data --visit_id 420683 --video_id 42445137 --dataset_assets hires_wide_intrinsics laser_scan_5mm hires_poses hires_wide hires_depth crop_mask annotations descriptions motions arkit_mesh transform
python -m data_downloader.data_asset_download --split custom --download_dir data --visit_id 420693 --video_id 42445173 --dataset_assets hires_wide_intrinsics laser_scan_5mm hires_poses hires_wide hires_depth crop_mask annotations descriptions motions arkit_mesh transform

SceneFun3D Toolkit
"""

import numpy as np
import tyro
from typing import Annotated
from utils.data_parser import DataParser
import viser
import time
import trimesh

def main(
  data_dir: Annotated[str, tyro.conf.arg(help="Path to the dataset")],
  visit_id: Annotated[str, tyro.conf.arg(help="Visit identifier")] = '420683',
  video_id: Annotated[str, tyro.conf.arg(help="Video sequence identifier")] = '42445137'):
  
  # Read required data assets
  dataParser = DataParser(data_dir)
  laser_point_cloud = dataParser.get_laser_scan(visit_id)
  points = np.array(laser_point_cloud.points) - np.mean(laser_point_cloud.points, axis=0)
  colors = np.array(laser_point_cloud.colors)
  
  arkit_mesh = dataParser.get_arkit_reconstruction(visit_id, video_id, format="mesh")
  transform = dataParser.get_transform(visit_id, video_id)
  arkit_mesh.transform(np.linalg.inv(transform))
  arkit_mesh.translate(-np.mean(laser_point_cloud.points, axis=0))
  vertices = np.asarray(arkit_mesh.vertices)
  faces = np.asarray(arkit_mesh.triangles)
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
  
  points = np.array(laser_point_cloud.points) - np.mean(laser_point_cloud.points, axis=0)
  colors = np.array(laser_point_cloud.colors)
  annotations = dataParser.get_annotations(visit_id)
  descriptions = dataParser.get_descriptions(visit_id)
  motions = dataParser.get_motions(visit_id)
  
  # Visualization
  server = viser.ViserServer()
  # server.scene.add_point_cloud(f"scene", points=points[::10], colors=colors[::10], point_size=0.01)
  server.scene.add_mesh_trimesh(name="/trimesh", mesh=mesh)
  
  annotation_positions = {}
  for i, annotation in enumerate(annotations):
    annot_id = annotation["annot_id"]
    label = annotation["label"]
    if label == "exclude":
      continue
    point_ids = annotation["indices"]
    annotation_positions[annot_id] = np.mean(points[point_ids], axis=0)
    server.scene.add_point_cloud(f"annotations/{i}/mask",
                                  points=points[point_ids],
                                  colors=colors[point_ids],
                                  point_size=0.01)
    server.scene.add_label(f"annotations/{i}/label",
                           label,
                           position=annotation_positions[annot_id])
  
  for i, description in enumerate(descriptions):
    annot_id = description["annot_id"][0]
    description_text = description["description"]
    server.scene.add_label(f"descriptions/{i}/description",
                           f'"{description_text}"',
                           position=annotation_positions[annot_id] - np.array([0.0, 0.0, 0.1]))
    
  for i, motion in enumerate(motions):
    annot_id = motion["annot_id"]
    motion_type = motion["motion_type"]
    motion_dir = motion["motion_dir"]
    motion_origin_idx = motion["motion_origin_idx"]
    motion_viz_orient = motion["motion_viz_orient"]
    if motion_type == "trans":  # translateion
      if motion_viz_orient == "inwards":
        segment = np.array([points[motion_origin_idx], points[motion_origin_idx] - (np.array(motion_dir) / 5.0)])  
        server.scene.add_line_segments(f"motions/{i}/inwards", points=np.expand_dims(segment, 0), colors=[0, 255, 0], line_width=5)
      elif motion_viz_orient == "outwards":
        segment = np.array([points[motion_origin_idx], points[motion_origin_idx] + (np.array(motion_dir) / 5.0)])  
        server.scene.add_line_segments(f"motions/{i}/outwards", points=np.expand_dims(segment, 0), colors=[255, 0, 0], line_width=5)
    elif motion_type == "rot":  # rotation
      vector1 = annotation_positions[annot_id] - points[motion_origin_idx]
      vector1_norm = np.linalg.norm(vector1)
      segments = np.array([
         [annotation_positions[annot_id], points[motion_origin_idx] + (np.array(motion_dir)) * vector1_norm*0.5],
         [annotation_positions[annot_id], points[motion_origin_idx] - (np.array(motion_dir)) * vector1_norm*0.5],
         [points[motion_origin_idx] + (np.array(motion_dir)) * vector1_norm * 0.5,
          points[motion_origin_idx] - (np.array(motion_dir)) * vector1_norm * 0.5],
        ])
      server.scene.add_line_segments(f"motions/{i}/outwards", points=segments, colors=[0, 0, 255], line_width=5)

  print('Done')
  while True:  # keep server alive
    time.sleep(0.2)

if __name__ == "__main__":
  tyro.cli(main)