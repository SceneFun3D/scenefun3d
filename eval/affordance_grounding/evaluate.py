import argparse
import os
import numpy as np
from eval.affordance_grounding.eval_utils.eval_script import evaluate
from eval.affordance_grounding.eval_utils.util_3d import get_excluded_point_mask
from utils.rle import rle_decode, rle_encode

def main(pred_dir, gt_dir):
    scene_desc_names = sorted(el[:-4] for el in os.listdir(gt_dir) if el.endswith('.txt'))
    scene_names = list(set(s.split('_')[0] for s in scene_desc_names))

    point_cloud_lengths = {}
    excluded_point_masks = {}
    for scene_name in scene_names:
        match = None
        for scene_desc_name in scene_desc_names:
            if scene_desc_name.startswith(scene_name + "_"):
                match = scene_desc_name
                break

        if match is not None:
            gt_mask = np.loadtxt(os.path.join(gt_dir, scene_desc_name+'.txt'), dtype=np.uint32)
            point_cloud_length = gt_mask.shape[0]
            point_cloud_lengths[scene_name] = point_cloud_length
            excluded_point_masks[scene_name] = get_excluded_point_mask(gt_mask)
        else:
            assert False, f'Could not find the scene {scene_name} as there is no match in the GT directory.'

    preds = {}
    for scene_desc_name in scene_desc_names[:]:
        point_cloud_length = point_cloud_lengths[scene_desc_name.split('_')[0]]
        # print(point_cloud_length)
        excluded_points_mask = excluded_point_masks[scene_desc_name.split('_')[0]]

        # Load predictions based on the requested submission format
        file_path = os.path.join(pred_dir, scene_desc_name+'.txt')  # {visit_id}_{desc_id}.txt file
        scene_pred_mask_list = np.loadtxt(file_path, dtype=str)  # (num_masks, 2)

        if scene_pred_mask_list.shape == (2,):
            scene_pred_mask_list = scene_pred_mask_list[np.newaxis, ...]

        assert scene_pred_mask_list.shape[1] == 2, 'Each line should have 2 values: instance mask path and confidence score.'
        pred_masks = []
        pred_scores = []
        for mask_path, conf_score in scene_pred_mask_list: 
            with open(os.path.join(pred_dir, mask_path), "r", encoding="utf-8") as file:
                rle_counts = file.read()
            
            pred_mask = rle_decode(counts = rle_counts, length=point_cloud_length)
            pred_mask = pred_mask[np.logical_not(excluded_points_mask)] 

            if np.sum(pred_mask) < 1: # means that the pred mask is empty after filtering out the prediction on the excluded points
                continue

            pred_masks.append(pred_mask)
            pred_scores.append(float(conf_score))

        assert len(pred_masks) == len(pred_scores), 'Number of masks and confidence scores should be the same.'
        
        preds[scene_desc_name] = {
            'pred_masks': np.vstack(pred_masks).T if len(pred_masks) > 0 else np.zeros((1, 1)),
            'pred_scores': np.vstack(pred_scores) if len(pred_masks) > 0 else np.zeros(1),
            'pred_classes': np.ones(len(pred_masks), dtype=np.int64) if len(pred_masks) > 0 else np.ones(1, dtype=np.int64)
        }

    ap_dict = evaluate(preds, gt_dir, print_result=True)
    del ap_dict['classes']
    # print(ap_dict)
 
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_dir",
        help="Specify the predictions directory. Predictions must be in the submission format, containing '<visit_id>_<desc_id>.txt' files and 'predicted_masks/' folder including all masks."
    )

    parser.add_argument(
        "--gt_dir",
        help="Specify the GT annotations directory. It must contain <visit_id>_<desc_id>.txt files for gt annotations, see https://github.com/OpenSun3D/cvpr24-challenge/blob/main/challenge_track_2/benchmark_data/gt_development_scenes"
    )

    args = parser.parse_args()

    main(args.pred_dir, args.gt_dir)
