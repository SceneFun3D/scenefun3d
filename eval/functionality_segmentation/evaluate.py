import argparse
import os
import numpy as np
from eval.functionality_segmentation.eval_utils.eval_script import evaluate
from eval.functionality_segmentation.eval_utils.util_3d import get_excluded_point_mask
from eval.functionality_segmentation.eval_utils.rle import rle_decode, rle_encode

def main(pred_dir, gt_dir):
    scene_names = sorted(el[:-4] for el in os.listdir(gt_dir) if el.endswith('.txt'))

    point_cloud_lengths = {}
    excluded_point_masks = {}
    for scene_name in scene_names:

        gt_mask = np.loadtxt(os.path.join(gt_dir, scene_name+'.txt'), dtype=np.uint32)
        point_cloud_length = gt_mask.shape[0]
        point_cloud_lengths[scene_name] = point_cloud_length
        excluded_point_masks[scene_name] = get_excluded_point_mask(gt_mask)

    preds = {}
    for scene_name in scene_names[:]:
        point_cloud_length = point_cloud_lengths[scene_name]
        excluded_points_mask = excluded_point_masks[scene_name]

        # Load predictions based on the requested submission format
        file_path = os.path.join(pred_dir, scene_name+'.txt')  # {visit_id}.txt file
        scene_pred_mask_list = np.loadtxt(file_path, dtype=str)  # (num_masks, 3)

        if scene_pred_mask_list.shape == (3,):
            scene_pred_mask_list = scene_pred_mask_list[np.newaxis, ...]

        assert scene_pred_mask_list.shape[1] == 3, 'Each line should have 3 values: instance mask path, affordance class and confidence score.'
        pred_masks = []
        pred_classes = []
        pred_scores = []
        for mask_path, affordance_class, conf_score in scene_pred_mask_list: 
            with open(os.path.join(pred_dir, mask_path), "r", encoding="utf-8") as file:
                rle_counts = file.read()
            
            pred_mask = rle_decode(counts = rle_counts, length=point_cloud_length)
            pred_mask = pred_mask[np.logical_not(excluded_points_mask)] 

            if np.sum(pred_mask) < 1: # means that the pred mask is empty after filtering out the prediction on the excluded points
                continue

            pred_masks.append(pred_mask)
            pred_classes.append(int(affordance_class))
            pred_scores.append(float(conf_score))

        assert len(pred_masks) == len(pred_scores) and len(pred_masks) == len(pred_classes), 'Number of masks, predicted affordance classes and confidence scores should be the same.'
        
        preds[scene_name] = {
            'pred_masks': np.vstack(pred_masks).T if len(pred_masks) > 0 else np.zeros((1, 1)),
            'pred_scores': np.vstack(pred_scores) if len(pred_masks) > 0 else np.zeros(1),
            'pred_classes':  np.vstack(pred_classes) if len(pred_masks) > 0 else np.zeros(1),
        }

    ap_dict = evaluate(preds, gt_dir, print_result=True)
    # print(ap_dict)
 
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_dir",
        help="Specify the predictions directory. Predictions must be in the submission format, containing '<visit_id>.txt' files and 'predicted_masks/' folder including all masks."
    )

    parser.add_argument(
        "--gt_dir",
        help="Specify the GT annotations directory. It must contain <visit_id>.txt files for GT annotations, see https://scenefun3d.github.io/documentation/benchmarks/task1/"
    )

    args = parser.parse_args()

    main(args.pred_dir, args.gt_dir)
