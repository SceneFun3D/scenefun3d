import argparse
import os
import numpy as np
from eval.functionality_segmentation.eval_utils.benchmark_labels \
    import CLASS_LABELS, VALID_CLASS_IDS, EXCLUDE_ID

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

        laser_scan_path = data_parser.get_data_asset_path(data_asset_identifier='laser_scan_5mm', visit_id=visit_id)
        plydata = PlyData.read(laser_scan_path)
        number_of_points = len(plydata['vertex'])

        # find excluded points
        excluded_points = []
        exclude_annotation_item = next((item for item in annotations if item["label"] == "exclude"), None)

        if exclude_annotation_item is not None:
            excluded_points = exclude_annotation_item["indices"]

        output_mask = np.zeros((number_of_points, 1), dtype=np.uint16)

        out_filename = f"{visit_id}.txt"
        out_filepath = os.path.join(out_gt_dir, out_filename)

        instance_id = 0
        for annotation_item in annotations:
            label = annotation_item['label']

            if label == "exclude":
                continue

            annot_indices = annotation_item['indices']
            class_id = VALID_CLASS_IDS[CLASS_LABELS.index(label)]
            output_mask[annot_indices] = class_id * 1000 + instance_id + 1

            instance_id += 1

        if excluded_points:
            output_mask[excluded_points, 0] = EXCLUDE_ID

        np.savetxt(out_filepath, output_mask, fmt='%i')

 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        help="Specify the directory where the data are stored."
    )

    parser.add_argument(
        "--val_scenes_list",
        help="Specify the filepath to a .txt file where each line contains a visit_id corresponding to a validation scene."
    )

    parser.add_argument(
        "--out_gt_dir",
        help="Specify the output directory where the GT annotations will be stored."
    )

    args = parser.parse_args()

    os.makedirs(args.out_gt_dir, exist_ok=True)

    main(args.data_dir, args.val_scenes_list, args.out_gt_dir)
