"""
Data Download helpers

SceneFun3D Toolkit

modified from https://github.com/apple/ARKitScenes/blob/main/download_data.py
"""

import subprocess
import pandas as pd
import os
from moviepy import *

ARKitScenes_url = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1'
SceneFun3D_url = 'https://cvg-data.inf.ethz.ch/scenefun3d/v1'
# SceneFun3D_url = 'https://cvg-data.inf.ethz.ch/scenefun3d/release_test'
TRAINING = 'Training'
VALIDATION = 'Validation'
HIGRES_DEPTH_ASSET_NAME = 'highres_depth'
POINT_CLOUDS_FOLDER = 'laser_scanner_point_clouds'

default_raw_dataset_assets = [
    'lowres_wide',
    'lowres_wide_intrinsics',
    'lowres_depth',
    'confidence',
    'lowres_poses',
    'vid_mov',
    'vid_mp4',
    'arkit_mesh',
    '3dod_annotation',
    # 'wide',
    # 'wide_intrinsics',
    # 'highres_depth',
    'vga_wide',
    'vga_wide_intrinsics',
    'ultrawide',
    'ultrawide_intrinsics',
    #### SceneFun3D assets
    'laser_scan_5mm',
    'crop_mask',
    'transform',
    'hires_wide',
    'hires_wide_intrinsics',
    'hires_depth',
    'hires_poses',
    'annotations',
    'descriptions',
    'motions'
]

scenefun3d_assets = [
    'laser_scan_5mm',
    'crop_mask',
    'transform',
    'hires_wide',
    'hires_depth',
    'hires_wide_intrinsics',
    'hires_poses',
    'annotations',
    'descriptions',
    'motions'
]

visit_related_assets = [
    'laser_scan_5mm',
    'crop_mask',
    'annotations',
    'descriptions',
    'motions'
]

missing_3dod_assets_video_ids = ['47334522', '47334523', '42897421', '45261582', '47333152', '47333155',
                                 '48458535', '48018733', '47429677', '48458541', '42897848', '47895482',
                                 '47333960', '47430089', '42899148', '42897612', '42899153', '42446164',
                                 '48018149', '47332198', '47334515', '45663223', '45663226', '45663227']

def visit_raw_files(visit_id, assets, metadata):
    file_names = []
    for asset in assets:
        if asset == 'laser_scan_5mm':
            file_names.append(f'{visit_id}_laser_scan.ply')
        elif asset == 'crop_mask':
            file_names.append(f'{visit_id}_crop_mask.npy')
        elif asset == 'annotations':
            file_names.append(f'{visit_id}_annotations.json')
        elif asset == 'descriptions':
            file_names.append(f'{visit_id}_descriptions.json')
        elif asset == 'motions':
            file_names.append(f'{visit_id}_motions.json')
        else:
            raise Exception(f'No asset = {asset} in raw dataset')
    return file_names

def video_raw_files(video_id, assets, metadata):
    file_names = []
    local_file_names = []
    for asset in assets:
        if HIGRES_DEPTH_ASSET_NAME == asset or asset == "wide" or asset == "wide_intrinsics":
            in_upsampling = metadata.loc[metadata['video_id'] == float(video_id), ['is_in_upsampling']].iat[0, 0]
            if not in_upsampling:
                print(f"Skipping asset {asset} for video_id {video_id} - Video not in upsampling dataset")
                continue  # highres_depth asset only available for video ids from upsampling dataset

        if asset in ['confidence', 'highres_depth', 'lowres_depth', 'lowres_wide', 'lowres_wide_intrinsics',
                     'ultrawide', 'ultrawide_intrinsics', 'vga_wide', 'vga_wide_intrinsics', 'wide', 'wide_intrinsics',
                     'hires_wide', 'hires_wide_intrinsics', 'hires_depth']:
            file_names.append(asset + '.zip')
            local_file_names.append(asset + '.zip')
        elif asset == 'vid_mov' or asset == 'vid_mp4':
            file_names.append(f'{video_id}.mov')
            local_file_names.append(f'{video_id}.mov')
        elif asset == 'arkit_mesh':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f'{video_id}_3dod_mesh.ply')
                local_file_names.append(f'{video_id}_arkit_mesh.ply')
        elif asset == '3dod_annotation':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f'{video_id}_3dod_annotation.json')
                local_file_names.append(f'{video_id}_3dod_annotation.json')
        elif asset == 'lowres_poses':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append('lowres_wide.traj')
                local_file_names.append('lowres_poses.traj')
        elif asset == 'hires_poses':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append('hires_poses.traj')
                local_file_names.append('hires_poses.traj')
        elif asset == 'transform':
            if video_id not in missing_3dod_assets_video_ids:
                file_names.append(f'{video_id}_refined_transform.npy')
                local_file_names.append(f'{video_id}_transform.npy')
        else:
            raise Exception(f'No asset = {asset} in raw dataset')
    return file_names, local_file_names


def download_file(url, file_name, dst):
    os.makedirs(dst, exist_ok=True)
    filepath = os.path.join(dst, file_name)

    if not os.path.isfile(filepath):
        command = f"curl {url} -o {file_name}.tmp --fail"
        print(f"Downloading file {filepath}")
        try:
            subprocess.check_call(command, shell=True, cwd=dst)
        except Exception as error:
            print(f'Error downloading {url}, error: {error}')
            return False
        os.rename(filepath+".tmp", filepath)
    else:
        print(f'WARNING: skipping download of existing file: {filepath}')
    return True


def unzip_file(file_name, dst, keep_zip=True):
    filepath = os.path.join(dst, file_name)
    print(f"Unzipping zip file {filepath}")
    command = f"unzip -oq {filepath} -d {dst}"
    try:
        subprocess.check_call(command, shell=True)
        # pass
    except Exception as error:
        print(f'Error unzipping {filepath}, error: {error}')
        return False
    if not keep_zip:
        os.remove(filepath)
        # pass
    return True

def download_assets_for_visit_id(visit_id, download_dir, split, dataset_assets):
    metadata = pd.read_csv("benchmark_file_lists/arkitscenes/metadata.csv")
    url_prefix = ""
    file_names = []
    if not dataset_assets:
        print(f"Warning: No assets given for visit id {visit_id}")
    else:
        dst_dir = download_dir #os.path.join(download_dir, str(visit_id))
        split_scenefun3d = "train" if split == "Training" else "test"
        url_prefix = f"{SceneFun3D_url}/{split_scenefun3d}/{visit_id}" + "/{}"
        file_names = visit_raw_files(visit_id, dataset_assets, metadata)

    for file_name in file_names:
        dst_path = os.path.join(dst_dir, file_name)
        url = url_prefix.format(file_name)

        download_file(url, dst_path, dst_dir)

def download_assets_for_video_id(visit_id, video_id, download_dir, split, dataset_assets):
    metadata = pd.read_csv("benchmark_file_lists/arkitscenes/metadata.csv")
    url_prefix = ""
    file_names = []
    local_file_names = []
    if not dataset_assets:
        print(f"Warning: No assets given for video id {video_id}")
    else:
        dst_dir = os.path.join(download_dir, str(video_id))
        
        file_names, local_file_names = video_raw_files(video_id, dataset_assets, metadata)

    for i, file_name in enumerate(file_names):
        local_file_name = local_file_names[i]
        dst_path = os.path.join(dst_dir, local_file_name)
        if dataset_assets[i] in scenefun3d_assets:
            split_scenefun3d = "train" if split == "Training" else "test"
            url_prefix = f"{SceneFun3D_url}/{split_scenefun3d}/{visit_id}/{video_id}" + "/{}"
        else:
            url_prefix = f"{ARKitScenes_url}/raw/{split}/{video_id}" + "/{}"
        url = url_prefix.format(file_name)

        if not file_name.endswith('.zip') or not os.path.isdir(dst_path[:-len('.zip')]):
            download_file(url, dst_path, dst_dir)
        else:
            print(f'WARNING: skipping download of existing zip file: {dst_path}')
        if local_file_name.endswith('.zip') and os.path.isfile(dst_path):
            unzip_file(local_file_name, dst_dir, keep_zip=False)

    if 'vid_mp4' in  dataset_assets:
        convert_mov_to_mp4(video_id, download_dir)

# Function to find the arkitscenes fold value for a given video_id
def find_arkitscenes_split(video_id):
    video_id = int(video_id)
    df = pd.read_csv('benchmark_file_lists/arkitscenes/metadata.csv')

    # Filter the DataFrame by the given video_id
    result = df[df['video_id'] == video_id]
    
    # Check if any matching row is found
    if not result.empty:
        # Return the fold value from the filtered result
        return result.iloc[0]['fold']
    else:
        assert False, f"No split was found for {video_id}"  # Return None if the video_id is not found

def convert_mov_to_mp4(video_id, 
                     download_dir):
    video_id = str(video_id)
    video_path = os.path.join(download_dir, video_id)

    print("Converting .mov to .mp4 ...")
    clip = VideoFileClip(os.path.join(video_path, f"{video_id}.mov"))

    clip.write_videofile(os.path.join(video_path, f"{video_id}.mp4"), audio=False, logger=None)

    os.remove(os.path.join(video_path, f"{video_id}.mov"))
