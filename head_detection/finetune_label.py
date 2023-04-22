import os
import random
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from ensemble_boxes import weighted_boxes_fusion
from utils import load_dict

def run_wbf(image_ids, box_pred, score_pred, label_pred, NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in image_ids:
        if image_id not in box_pred.keys(): continue
        boxes = box_pred[image_id]
        scores = score_pred[image_id]
        labels = label_pred[image_id]

        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=NMS_THRESH, skip_box_thr=BOX_THRESH)
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]
        labels = labels[idxs]

        if len(boxes) > 0:
            height, width = 1080, 1920
            boxes[:, [0,2]] = (boxes[:, [0,2]]*width).clip(min=0, max=width-1)
            boxes[:, [1,3]] = (boxes[:, [1,3]]*height).clip(min=0, max=height-1)
        output_dict[image_id] = (boxes, scores, labels)
    return output_dict

def emsemble_obj(output_txt_file, image_ids):
    effdet_ed7_768_box_pred = load_dict('effdet_ed7_768_box_pred.pkl')
    effdet_ed7_768_score_pred = load_dict('effdet_ed7_768_score_pred.pkl')
    effdet_ed7_768_label_pred = load_dict('effdet_ed7_768_label_pred.pkl')
    
    output_dict = run_wbf(image_ids, effdet_ed7_768_box_pred, effdet_ed7_768_score_pred,
                          effdet_ed7_768_label_pred, NMS_THRESH=0.5, BOX_THRESH=0.1, PP_THRESH=0.1)
    
    for id, obj in output_dict.items():
        bboxs, scores, labels = obj[0], obj[1], obj[2]
        for idx, bbox in enumerate(bboxs):
            video_id = id.split("_")[0]
            frame_id = id.split("_")[1]
            content_line = '{},{},{},{},{},{},{},{}'.format(int(video_id), int(frame_id), int(bbox[0]),
                                                        int(bbox[1]), int(bbox[2]) - int(bbox[0]),
                                                        int(bbox[3]) - int(bbox[1]), int(labels[idx]),
                                                        scores[idx])
            print(content_line)
            output_txt_file.write(content_line)
            output_txt_file.write('\n')


if __name__ == '__main__':
    df = '../aicity_dataset/aicity2023_track5_test_images/'
    image_file_list = [f for f in listdir(df) if isfile(join(df, f))]
    image_ids = []
    for id in image_file_list:
        image_ids.append(id.split(".")[0])
    txt_path = 'effdet_ed7_768_head.txt'
    output_txt_file = open(txt_path, "w")
    emsemble_obj(output_txt_file, image_ids)
    