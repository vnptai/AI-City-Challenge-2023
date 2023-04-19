import pandas as pd
import numpy as np
import numba
import re
import cv2
import ast
import matplotlib.pyplot as plt

from numba import jit
from typing import List, Union, Tuple

# Numba typed list!
iou_thresholds = numba.typed.List()
for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    iou_thresholds.append(x)

# @jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


# @jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

# @jit(nopython=True)
def calculate_precision(gts, preds, preds_labels, threshold = 0.5, form = 'coco', ious=None, num_class=7) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    ap_all = {}
    for i in range(num_class + 1):
        ap_all[i] = {}
        ap_all[i]['tp'] = 0
        ap_all[i]['fp'] = 0
        ap_all[i]['fn'] = 0
        # ap_all[i]['fp'] = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):
        gts_idx = gts[gts[:, 4] == preds_labels[pred_idx]]
        best_match_gt_idx = find_best_match(gts_idx, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            # tp += 1
            ap_all[preds_labels[pred_idx]]['tp'] += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            # fp += 1
            ap_all[preds_labels[pred_idx]]['fp'] += 1

    # False negative: indicates a gt box had no associated predicted box.
    for gt in gts:
       if gt[-1] > 0:
           class_idx_miss = gt[-1]
           ap_all[int(class_idx_miss)]['fn'] += 1
    # fn = (gts.sum(axis=1) > 0).sum()

    for i in range(num_class):
        ap_all[0]['tp'] += ap_all[i]['tp']
        ap_all[0]['fp'] += ap_all[i]['fp']
        ap_all[0]['fn'] += ap_all[i]['fn']
    # ap_all[0]['tp'] = ap_all[0]['tp'] / num_class
    # ap_all[0]['fp'] = ap_all[0]['fp'] / num_class
    # ap_all[0]['fn'] = ap_all[0]['fn'] / num_class

    for key, value in ap_all.items():
        if ap_all[key]['tp'] == 0 and (ap_all[key]['tp'] + ap_all[key]['fp'] + ap_all[key]['fn']) == 0:
            ap_all[key]['ap'] = 1.0
        else:
            ap_all[key]['ap'] = ap_all[key]['tp'] / (ap_all[key]['tp'] + ap_all[key]['fp'] + ap_all[key]['fn'])

    return ap_all
    # if tp == 0 and (tp + fp + fn) == 0:
    #     return 1.0
    #
    # return tp / (tp + fp + fn)


# @jit(nopython=True)
def calculate_image_precision(gts, preds, labels, thresholds = (0.5, ), form = 'coco', num_class=7) -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    # gt_labels = gts.copy()[-1]
    # gts = gts[:4]
    ap_thres = {}
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        ap_thres[str(threshold)] = {}
        for i in range(num_class + 1):
            ap_thres[str(threshold)][i] = 0
    ap_thres['all'] = {}
    for i in range(num_class + 1):
        ap_thres['all'][i] = 0

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, labels, threshold=threshold,
                                                     form=form, ious=ious, num_class=num_class)
        for key, value in precision_at_threshold.items():
            ap_thres[str(threshold)][key] = value['ap']
        # ap_thres[str(threshold)]['ap'] = precision_at_threshold[0]['ap']
            ap_thres['all'][key] += precision_at_threshold[key]['ap']
        # image_precision += precision_at_threshold / n_threshold

    for key, value in ap_thres['all'].items():
        ap_thres['all'][key] = value / n_threshold
    # return image_precision
    return ap_thres

def calculate_final_score(all_predictions, score_threshold, num_class=7):
    # final_scores = []
    final_scores = {}
    for thres_hold in iou_thresholds:
        final_scores[str(thres_hold)] = {}
        for i in range(num_class + 1):
            final_scores[str(thres_hold)][i] = 0
    final_scores['all'] = {}
    for i in range(num_class + 1):
        final_scores['all'][i] = 0
    for i in range(len(all_predictions)):
        gt_boxes = all_predictions[i]['gt_boxes'].copy()
        pred_boxes = all_predictions[i]['pred_boxes'].copy()
        scores = all_predictions[i]['scores'].copy()
        labels = all_predictions[i]['labels'].copy()

        indexes = np.where(scores > score_threshold)
        pred_boxes = pred_boxes[indexes]
        scores = scores[indexes]
        pred_labels = labels[indexes]

        image_precision = calculate_image_precision(gt_boxes, pred_boxes, pred_labels, thresholds=iou_thresholds,form='pascal_voc')
        # final_scores.append(image_precision)
        for key, value in image_precision.items():
            for key1, value1 in value.items():
                final_scores[key][key1] += value1

    for key, value in final_scores.items():
        for key1, value1 in value.items():
            final_scores[key][key1] = value1/len(all_predictions)

    # return np.mean(final_scores)
    return final_scores