import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from utils import get_resolution, save_dict, load_dict, format_prediction_string, make_pseudo_dataframe, refine_checkpoint_in, refine_checkpoint_out
from os import listdir
from os.path import isfile, join
import pandas as pd
def run_wbf_4preds(image_ids,
                   box_pred1, score_pred1, label_pred1,
                   box_pred2, score_pred2, label_pred2,
                    box_pred3, score_pred3, label_pred3,
                    box_pred4, score_pred4, label_pred4,
                   NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in image_ids:
        if image_id not in box_pred1.keys(): continue
        boxes = box_pred1[image_id] + box_pred2[image_id] + box_pred3[image_id] + box_pred4[image_id]
        scores = score_pred1[image_id] + score_pred2[image_id] + score_pred3[image_id] + score_pred4[image_id]
        labels = label_pred1[image_id] + label_pred2[image_id] + label_pred3[image_id] + label_pred4[image_id]

        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=NMS_THRESH, skip_box_thr=BOX_THRESH)
        boxes = np.array(boxes)
        scores = np.array(scores)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            height, width = 1080, 1920
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(min=0, max=width - 1)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(min=0, max=height - 1)
        output_dict[image_id] = (boxes, scores, labels)
    return output_dict

effdet_ed7_896_box_pred = load_dict('pkl/effdet_ed7_896_box_pred.pkl')
effdet_ed7_896_score_pred = load_dict('pkl/effdet_ed7_896_score_pred.pkl')
effdet_ed7_896_label_pred = load_dict('pkl/effdet_ed7_896_label_pred.pkl')

effdet_ed6_768_box_pred = load_dict('pkl/effdet_ed6_768_box_pred.pkl')
effdet_ed6_768_score_pred = load_dict('pkl/effdet_ed6_768_score_pred.pkl')
effdet_ed6_768_label_pred = load_dict('pkl/effdet_ed6_768_label_pred.pkl')

effdet_ed5_768_box_pred = load_dict('pkl/effdet_ed5_768_box_pred.pkl')
effdet_ed5_768_score_pred = load_dict('pkl/effdet_ed5_768_score_pred.pkl')
effdet_ed5_768_label_pred = load_dict('pkl/effdet_ed5_768_label_pred.pkl')

effdet_ed7_1024_box_pred = load_dict('pkl/effdet_ed7_1024_box_pred.pkl')
effdet_ed7_1024_score_pred = load_dict('pkl/effdet_ed7_1024_score_pred.pkl')
effdet_ed7_1024_label_pred = load_dict('pkl/effdet_ed7_1024_label_pred.pkl')

df = '../aicity_dataset/aicity2023_track5_images/'
image_file_list = [f for f in listdir(df) if isfile(join(df, f))]
image_ids = []
for id in image_file_list:
    image_ids.append(id.split(".")[0])

TEST_DIR = '../aicity_dataset/aicity2023_track5_test_images'
TRAIN_DIR = '../aicity_dataset/aicity2023_track5_images'

output_dict = run_wbf_4preds(image_ids,
                         effdet_ed7_896_box_pred, effdet_ed7_896_score_pred, effdet_ed7_896_label_pred,
                         effdet_ed6_768_box_pred, effdet_ed6_768_score_pred, effdet_ed6_768_label_pred,
                         effdet_ed5_768_box_pred, effdet_ed5_768_score_pred, effdet_ed5_768_label_pred,
                         effdet_ed7_1024_box_pred, effdet_ed7_1024_score_pred, effdet_ed7_1024_label_pred,
                         NMS_THRESH=0.50, BOX_THRESH=0.32, PP_THRESH=0.26)

df = pd.read_csv('dataset/trainset_ai.csv')
PSEUDO_FOLD = 1
make_pseudo_dataframe(image_ids, output_dict, TEST_DIR, df, TRAIN_DIR, PSEUDO_FOLD)

