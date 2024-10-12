# Adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py

import os, sys
from copy import deepcopy
from uuid import uuid4
import numpy as np
from scipy import stats
from tqdm import tqdm
from eval.affordance_grounding.eval_utils import util
from eval.affordance_grounding.eval_utils import util_3d
from eval.affordance_grounding.eval_utils.benchmark_labels \
    import CLASS_LABELS, VALID_CLASS_IDS

# ---------- Label info ---------- #
CLASS_LABELS = ['functional_element']
VALID_CLASS_IDS = np.array([1])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

MIN_REGION_SIZE = 1 # 100 for s3dis, scannet
# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt['overlaps'] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt['min_region_sizes'] = np.array([MIN_REGION_SIZE])  # 100 for s3dis, scannet
# distance thresholds [m]
opt['distance_threshes'] = np.array([float('inf')])
# distance confidences
opt['distance_confs'] = np.array([-float('inf')])

def evaluate_matches(matches):
    overlaps = opt['overlaps']
    min_region_sizes = [opt['min_region_sizes'][0]]
    dist_threshes = [opt['distance_threshes'][0]]
    dist_confs = [opt['distance_confs'][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'uuid' in p:
                                pred_visited[p['uuid']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt[
                                        'med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['uuid']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                        gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['uuid']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                                        gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            cur_true = np.append(cur_true, 0)
                            confidence = pred["confidence"]
                            cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps, factor=100): # factor is 1 or 100 (for percent)
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25]) * factor
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50]) * factor
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25]) * factor
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25]) * factor
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50]) * factor
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25]) * factor
    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
    for i in range(len(pred['pred_classes'])):
        info = {}
        info["label_id"] = pred['pred_classes'][i]
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def assign_instances_for_scan(pred: dict, gt_file: str):
    pred_info = make_pred_info(pred)
    try:
        gt_ids = util_3d.load_ids(gt_file)
        # breakpoint()
        exclude_mask = util_3d.get_excluded_point_mask(gt_ids)
        gt_ids = gt_ids[np.logical_not(exclude_mask)]
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # breakpoint()
    # go thru all prediction masks
    for uuid in pred_info:
        label_id = int(pred_info[uuid]['label_id'])
        conf = pred_info[uuid]['conf']
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]['mask']
        assert (len(pred_mask) == len(gt_ids))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt['min_region_sizes'][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['uuid'] = uuid
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    # print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')


def fast_hist(pred, gt, num_labels):
    k = (gt >= 0) & (gt < num_labels)
    return np.bincount(num_labels * gt[k].astype(int) + pred[k], minlength=num_labels**2).reshape(num_labels, num_labels)

def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def evaluate(preds: dict, gt_path: str, print_result=True):
    matches = {}
    for (k, v) in tqdm(preds.items(), desc="evaluate instances"):
        gt_file = os.path.join(gt_path, k + ".txt")
        if not os.path.isfile(gt_file):
            util.print_error('Scan {} does not match any gt file'.format(k), user_fault=True)

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(v, gt_file)
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    if print_result:
        print_results(avgs)

    return avgs


