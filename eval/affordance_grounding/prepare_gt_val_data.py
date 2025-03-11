"""
This script reads scene annotations from the validation set and 
organizes them into the evaluation ground truth (GT) format.
"""

import argparse
import os
import numpy as np
from eval.affordance_grounding.eval_utils.benchmark_labels \
    import EXCLUDE_ID

from tqdm import tqdm
from utils.data_parser import DataParser
from plyfile import PlyData

def main(data_dir, val_scenes_list, out_gt_dir):

    data_parser = DataParser(data_root_path=data_dir)

    with open(val_scenes_list, 'r') as file:
        scene_id_list = file.readlines()

    scene_id_list = [scene_id.strip() for scene_id in scene_id_list]

    for scene_id in tqdm(scene_id_list, desc="Extracting GT data"):
        visit_id = scene_id
        annotations = data_parser.get_annotations(visit_id, group_excluded_points=True)
        descriptions = data_parser.get_descriptions(visit_id)

        laser_scan_path = data_parser.get_data_asset_path(data_asset_identifier='laser_scan_5mm', visit_id=visit_id)
        plydata = PlyData.read(laser_scan_path)
        number_of_points = len(plydata['vertex'])

        # find excluded points
        excluded_points = []
        exclude_annotation_item = next((item for item in annotations if item["label"] == "exclude"), None)

        if exclude_annotation_item is not None:
            excluded_points = exclude_annotation_item["indices"]

        for description_item in descriptions:
            desc_id = description_item['desc_id']
            annot_id_list = description_item['annot_id']

            out_filename = f"{visit_id}_{desc_id}.txt"
            out_filepath = os.path.join(out_gt_dir, out_filename)

            output_masks = np.zeros((number_of_points, 1), dtype=np.uint8)

            if excluded_points:
                output_masks[excluded_points, 0] = EXCLUDE_ID

            for i, cur_annot_id in enumerate(annot_id_list):
                annotation_item = next((item for item in annotations if item["annot_id"] == cur_annot_id), None)

                assert annotation_item is not None, "Retrieved annotation item must not be None"

                annotation_indices = annotation_item['indices']

                output_masks[annotation_indices, 0] = i+1

            np.savetxt(out_filepath, output_masks, fmt='%i')

 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        help="Specify the predictions directory. Predictions must be in the submission format, containing '<visit_id>_<desc_id>.txt' files and 'predicted_masks/' folder including all masks."
    )

    parser.add_argument(
        "--val_scenes_list",
        help="Specify the predictions directory. Predictions must be in the submission format, containing '<visit_id>_<desc_id>.txt' files and 'predicted_masks/' folder including all masks."
    )

    parser.add_argument(
        "--out_gt_dir",
        help="Specify the GT annotations directory. It must contain <visit_id>_<desc_id>.txt files for gt annotations, see https://github.com/OpenSun3D/cvpr24-challenge/blob/main/challenge_track_2/benchmark_data/gt_development_scenes"
    )

    args = parser.parse_args()

    os.makedirs(args.out_gt_dir, exist_ok=True)

    main(args.data_dir, args.val_scenes_list, args.out_gt_dir)
