"""
Downloads the data assets in the SceneFun3D dataset

SceneFun3D Toolkit

"""

import argparse
import os
import pandas as pd
from data_downloader.download_utils.download_data import download_assets_for_visit_id, \
    download_assets_for_video_id, default_raw_dataset_assets, find_arkitscenes_split, \
    visit_related_assets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train_val_set", "test_set", "custom"],
        help="Specify the split of the data. This argument can be 'train_val_set', 'test_set' or 'custom'."
    )

    parser.add_argument(
        "--download_dir",
        default="data",
        help="Specify the path where the downloaded data will be stored."
    )

    parser.add_argument(
        "--download_only_one_video_sequence",
        action="store_true",
        help= (
            "(Optional) Specify whether to download only one video sequence "
            "(the longest video will be downloaded). By omitting this flag, all the "
            "available video sequences for each scene will be downloaded."
        )
    )

    parser.add_argument(
        "--dataset_assets",
        required=True,
        nargs='+',
        choices=default_raw_dataset_assets,
        help="Specify the identifier list of the data assets to download"
    )

    parser.add_argument(
        "--visit_id",
        required=False,
        help="Specify the visit_id of the scene to download assets for. Applicable only when split is set to 'custom'."
    )

    parser.add_argument(
        "--video_id",
        required=False,
        help="Specify the video_id of the scene to download assets for. Applicable only when split is set to 'custom'."
    )

    parser.add_argument(
        "--video_id_csv",
        required=False,
        help="Specify the .csv filepath which contains the pairs of visit_id-video_id to download assets for. Applicable only when split is set to 'custom'."
    )

    args = parser.parse_args()

    video_id_csv = None
    df = None
    if args.split == "train_val_set" and args.download_only_one_video_sequence:
        video_id_csv = "benchmark_file_lists/train_val_set_only_one_video.csv"
    elif args.split == "train_val_set":
        video_id_csv = "benchmark_file_lists/train_val_set.csv"
    elif args.split == "test_set" and args.download_only_one_video_sequence:
        video_id_csv = "benchmark_file_lists/test_set_only_one_video.csv"
    elif args.split == "test_set":
        video_id_csv = "benchmark_file_lists/test_set.csv"
    elif args.split == "custom":
        if args.video_id_csv:
            video_id_csv = args.video_id_csv
        elif (args.visit_id and args.video_id):
            df = pd.DataFrame({
                'visit_id': [args.visit_id],
                'video_id': [args.video_id]
            })
        else:
            assert False, "For split 'custom', either a) video_id_csv must be provided or b) visit_id and video_id must be provided"

    if df is None:
        df = pd.read_csv(video_id_csv)

    download_dir = os.path.abspath(args.download_dir)
    assets_folder_path = download_dir
    os.makedirs(assets_folder_path, exist_ok=True)

    visit_dataset_assets = []
    video_dataset_assets = []
    for asset in args.dataset_assets:
        if asset in visit_related_assets:
            visit_dataset_assets.append(asset)
        else:
            video_dataset_assets.append(asset)

    number_of_rows = len(df.index)

    # iterate through each visit_id - video_id 
    prev_visit_id_list = []
    for index, row in df.iterrows():
        visit_id = int(row['visit_id'])
        video_id = int(row['video_id'])

        print(f"[*] Downloading assets - visit_id: {visit_id}, video_id: {video_id} (progress: {index} / {number_of_rows})")

        split = find_arkitscenes_split(video_id)

        current_asset_download_dir = os.path.join(assets_folder_path, str(visit_id))

        if visit_dataset_assets and str(visit_id) not in prev_visit_id_list:
            download_assets_for_visit_id(str(visit_id), current_asset_download_dir, split, visit_dataset_assets)

        prev_visit_id_list.append(str(visit_id))
        
        if video_dataset_assets:
            download_assets_for_video_id(str(visit_id), video_id, current_asset_download_dir, split, video_dataset_assets)
            
        print("\n")

    print("Done.")

