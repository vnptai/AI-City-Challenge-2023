import os
import random
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sort import *
from ensemble_boxes import weighted_boxes_fusion
from utils import load_dict

AICITY_CLASSES = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet']
HEAD_CLASSES = ['head', 'helmet', 'uncertain']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(8)]

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

def run_wbf_6preds(image_ids,
                   box_pred1, score_pred1, label_pred1,
                   box_pred2, score_pred2, label_pred2,
                    box_pred3, score_pred3, label_pred3,
                    box_pred4, score_pred4, label_pred4,
                    box_pred5, score_pred5, label_pred5,
                    box_pred6, score_pred6, label_pred6,
                   NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in image_ids:
        if image_id not in box_pred1.keys(): continue
        boxes = box_pred1[image_id] + box_pred2[image_id] + box_pred3[image_id] + box_pred4[image_id] + box_pred5[image_id] + box_pred6[image_id]
        scores = score_pred1[image_id] + score_pred2[image_id] + score_pred3[image_id] + score_pred4[image_id] + score_pred5[image_id] + score_pred6[image_id]
        labels = label_pred1[image_id] + label_pred2[image_id] + label_pred3[image_id] + label_pred4[image_id] + label_pred5[image_id] + label_pred6[image_id]

        list_box = []
        list_score = []
        list_label = []
        for box in boxes:
            for b in box:
                list_box.append(b)
        for score in scores:
            for s in score:
                list_score.append(s)
        for label in labels:
            for l in label:
                list_label.append(l)

        boxes = np.array(list_box)
        scores = np.array(list_score)
        labels = np.array(list_label)
        remove_idx = []
        for idx, label in enumerate(list_label):
            if int(label) == 5: continue
            remove_idx.append(idx)
        boxes = np.delete(boxes, remove_idx, axis=0)
        scores = np.delete(scores, remove_idx, axis=0)
        labels = np.delete(labels, remove_idx, axis=0)
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=NMS_THRESH,
                                                      skip_box_thr=BOX_THRESH)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            height, width = 1080, 1920
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(min=0, max=width - 1)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(min=0, max=height - 1)
        output_dict[image_id] = (boxes, scores, labels)


        # boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=NMS_THRESH, skip_box_thr=BOX_THRESH)
        # boxes = np.array(boxes)
        # scores = np.array(scores)
        #
        # idxs = np.where(scores > PP_THRESH)[0]
        # boxes = boxes[idxs]
        # scores = scores[idxs]
        #
        # if len(boxes) > 0:
        #     height, width = 1080, 1920
        #     boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(min=0, max=width - 1)
        #     boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(min=0, max=height - 1)
        # output_dict[image_id] = (boxes, scores, labels)
    return output_dict

def run_wbf_2preds(image_ids,
                   box_pred1, score_pred1, label_pred1,
                   box_pred2, score_pred2, label_pred2,
                   NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in image_ids:
        if image_id not in box_pred1.keys(): continue
        boxes = box_pred1[image_id] + box_pred2[image_id]
        scores = score_pred1[image_id] + score_pred2[image_id]
        labels = label_pred1[image_id] + label_pred2[image_id]

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

def run_wbf(image_ids, box_pred, score_pred, label_pred, NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in image_ids:
        if image_id not in box_pred.keys(): continue

        boxes = box_pred[image_id]
        scores = score_pred[image_id]
        labels = label_pred[image_id]
        list_box = []
        list_score = []
        list_label = []
        for box in boxes:
            for b in box:
                list_box.append(b)
        for score in scores:
            for s in score:
                list_score.append(s)
        for label in labels:
            for l in label:
                list_label.append(l)

        boxes = np.array(list_box)
        scores = np.array(list_score)
        labels = np.array(list_label)
        remove_idx = []
        for idx, label in enumerate(list_label):
            if int(label) == 5: continue
            remove_idx.append(idx)
        boxes = np.delete(boxes, remove_idx, axis=0)
        scores = np.delete(scores, remove_idx, axis=0)
        labels = np.delete(labels, remove_idx, axis=0)
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=NMS_THRESH,
                                                      skip_box_thr=BOX_THRESH)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            height, width = 1080, 1920
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(min=0, max=width - 1)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(min=0, max=height - 1)
        output_dict[image_id] = (boxes, scores, labels)

        # boxes = box_pred[image_id]
        # scores = score_pred[image_id]
        # labels = label_pred[image_id]
        #
        # boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=NMS_THRESH, skip_box_thr=BOX_THRESH)
        # boxes = np.array(boxes)
        # scores = np.array(scores)
        # labels = np.array(labels)
        #
        # idxs = np.where(scores > PP_THRESH)[0]
        # boxes = boxes[idxs]
        # scores = scores[idxs]
        # labels = labels[idxs]
        #
        # if len(boxes) > 0:
        #     height, width = 1080, 1920
        #     boxes[:, [0,2]] = (boxes[:, [0,2]]*width).clip(min=0, max=width-1)
        #     boxes[:, [1,3]] = (boxes[:, [1,3]]*height).clip(min=0, max=height-1)
        # output_dict[image_id] = (boxes, scores, labels)
    return output_dict

def overlap_ratio(boxA, boxB):
    """Calculate iou of 2 bboxes

    Args:
        boxA (list): [x1, y1, x2, y2]
        boxB (list): [x1, y1, x2, y2]

    Returns:
        overlap_ratio (float): overlap ratio max(interArea / float(boxBArea), interArea / float(boxAArea))
    """
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if boxBArea == 0 or boxAArea == 0:
        return 0
    overlap_ratio = max(interArea / float(boxBArea), interArea / float(boxAArea))
    # return the overlap ratio
    return overlap_ratio

def plot_one_box(x, image, color=None, label=None, line_thickness=True):
    # Plots one bounding box on image img
    # tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    # print(tl)
    tl = 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_head_box(path):
    head_bbox_info = {}
    f = open(path, "r")
    for line in f.readlines():
        data = line.split(',')
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, class_confidence = int(data[0]), int(data[1]), \
            int(data[2]), int(data[3]), int(data[4]), int(data[5]), int(data[6]), float(data[7].split('\n')[0])
        if video_id not in head_bbox_info.keys():
            head_bbox_info[video_id] = {}
        else:
            if frame not in head_bbox_info[video_id].keys():
                head_bbox_info[video_id][frame] = []
            else:
                head_bbox_info[video_id][frame].append(
                    [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, class_id, class_confidence])

    return head_bbox_info

def visualize_image(image_path, obj_box, head_box):
    image = cv2.imread(image_path)

    plot_one_box([head_box[0], head_box[1], head_box[2], head_box[3]], image, color=colors[head_box[4]],
                     label=HEAD_CLASSES[head_box[4]-1], line_thickness=True)
    plot_one_box([obj_box[0], obj_box[1], obj_box[2], obj_box[3]], image, color=colors[obj_box[4]],
                 label=AICITY_CLASSES[obj_box[4] - 1], line_thickness=True)
    cv2.imshow(image_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_original_box(path):
    f = open(path, "r")
    original_bbox_info = {}
    for line in f.readlines():
        data = line.split(',')
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, class_confidence = int(data[0]), int(data[1]), \
            int(data[2]), int(data[3]), int(data[4]), int(data[5]), int(data[6]), float(data[7].split('\n')[0])
        if video_id not in original_bbox_info.keys():
            original_bbox_info[video_id] = {}
        if frame not in original_bbox_info[video_id].keys():
            original_bbox_info[video_id][frame] = []
        original_bbox_info[video_id][frame].append([bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, class_id, class_confidence])
    return original_bbox_info

def finetune_listvideo(video_file_list, original_bbox_info, output_txt_file, overlap_thres=0.95):
    for video_file in video_file_list:
        video_path = video_file_path + video_file
        video_number = int(video_file.split(".")[0])
        cap = cv2.VideoCapture(video_path)
        index_frame = 0

        size = (1920, 1080)
        writer = cv2.VideoWriter(video_file_path_out + video_file,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret != True: break
            index_frame += 1
            if index_frame in original_bbox_info[video_number].keys():
                object_list = original_bbox_info[video_number][index_frame]

                for object in object_list:
                    object.append(video_number)
                    object.append(index_frame)

                object_list = np.array(object_list)

                for obj in object_list:
                    content_line = '{},{},{},{},{},{},{},{}'.format(video_number, index_frame, int(obj[0]),
                                                                    int(obj[1]), int(obj[2]) - int(obj[0]),
                                                                    int(obj[3]) - int(obj[1]), int(obj[4]),
                                                                    obj[5])
                    # output_txt_file.write(content_line)
                    # output_txt_file.write('\n')
                    plot_one_box([int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])], frame, color=colors[int(obj[4])],
                                 label=str(int(obj[4])), line_thickness=True)
            # writer.write(frame)
            cv2.imshow('image', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("The video " + video_file_path_out + video_file + " was successfully saved")

def fine_tune_obj(original_bbox_info, output_txt_file, overlap_thres=0.95):
    for video_id, frame_info in original_bbox_info.items():
        mot_tracker = Sort()
        for frame_id, object_list in frame_info.items():
            remove_indexes = []
            num_vehicle = len(object_list)

            for object in object_list:
                object.append(video_id)
                object.append(frame_id)

            object_list = np.array(object_list)

            ###update tracker
            trackers = mot_tracker.update(object_list)

            if len(trackers) > 0:
                for trk in trackers:
                    history_bbox = trk.votting()
                    for bbox in history_bbox:
                        content_line = '{},{},{},{},{},{},{},{}'.format(int(bbox[6]), int(bbox[7]), int(bbox[0]),
                                                                        int(bbox[1]), int(bbox[2]) - int(bbox[0]),
                                                                        int(bbox[3]) - int(bbox[1]), int(bbox[4]),
                                                                        bbox[5])
                        # print(content_line)
                        output_txt_file.write(content_line)
                        output_txt_file.write('\n')

            # for obj in new_objects:
            #     content_line = '{},{},{},{},{},{},{},{}'.format(video_id, frame_id, int(obj[0]),
            #                                                     int(obj[1]), int(obj[2]) - int(obj[0]),
            #                                                     int(obj[3]) - int(obj[1]), int(obj[4]),
            #                                                     obj[5])
            #     output_txt_file.write(content_line)
            #     output_txt_file.write('\n')
        print("The video " + str(video_id) + " was successfully saved")

def emsemble_obj(output_txt_file, image_ids):
    effdet_ed7_896_box_pred = load_dict('pkl_new/effdet_ed7_896_box_pred.pkl')
    effdet_ed7_896_score_pred = load_dict('pkl_new/effdet_ed7_896_score_pred.pkl')
    effdet_ed7_896_label_pred = load_dict('pkl_new/effdet_ed7_896_label_pred.pkl')

    effdet_ed6_768_box_pred = load_dict('pkl_new/effdet_ed6_768_box_pred.pkl')
    effdet_ed6_768_score_pred = load_dict('pkl_new/effdet_ed6_768_score_pred.pkl')
    effdet_ed6_768_label_pred = load_dict('pkl_new/effdet_ed6_768_label_pred.pkl')

    effdet_ed5_768_box_pred = load_dict('pkl_new/effdet_ed5_768_box_pred.pkl')
    effdet_ed5_768_score_pred = load_dict('pkl_new/effdet_ed5_768_score_pred.pkl')
    effdet_ed5_768_label_pred = load_dict('pkl_new/effdet_ed5_768_label_pred.pkl')

    effdet_ed6_640_box_pred = load_dict('effdet_ed6_640_box_pred.pkl')
    effdet_ed6_640_score_pred = load_dict('effdet_ed6_640_score_pred.pkl')
    effdet_ed6_640_label_pred = load_dict('effdet_ed6_640_label_pred.pkl')

    effdet_ed7_1024_box_pred = load_dict('pkl_new/effdet_ed7_1024_box_pred.pkl')
    effdet_ed7_1024_score_pred = load_dict('pkl_new/effdet_ed7_1024_score_pred.pkl')
    effdet_ed7_1024_label_pred = load_dict('pkl_new/effdet_ed7_1024_label_pred.pkl')

    effdet_ed7_768_box_pred = load_dict('pkl_new/effdet_ed7_768_box_pred.pkl')
    effdet_ed7_768_score_pred = load_dict('pkl_new/effdet_ed7_768_score_pred.pkl')
    effdet_ed7_768_label_pred = load_dict('pkl_new/effdet_ed7_768_label_pred.pkl')

    # output_dict = run_wbf_6preds(image_ids, effdet_ed7_896_box_pred, effdet_ed7_896_score_pred,
    #                       effdet_ed7_896_label_pred, effdet_ed6_768_box_pred, effdet_ed6_768_score_pred,
    #                       effdet_ed6_768_label_pred, effdet_ed5_768_box_pred, effdet_ed5_768_score_pred,
    #                       effdet_ed5_768_label_pred, effdet_ed6_640_box_pred, effdet_ed6_640_score_pred,
    #                       effdet_ed6_640_label_pred, effdet_ed7_1024_box_pred, effdet_ed7_1024_score_pred,
    #                       effdet_ed7_1024_label_pred, effdet_ed7_768_box_pred, effdet_ed7_768_score_pred,
    #                       effdet_ed7_768_label_pred, NMS_THRESH=0.5, BOX_THRESH=0.44, PP_THRESH=0.45)

    output_dict = run_wbf(image_ids, effdet_ed6_640_box_pred, effdet_ed6_640_score_pred,
                          effdet_ed6_640_label_pred, NMS_THRESH=0.5, BOX_THRESH=0.3, PP_THRESH=0.3)

    for id, obj in output_dict.items():
        bboxs, scores, labels = obj[0], obj[1], obj[2]
        for idx, bbox in enumerate(bboxs):
            video_id = id.split("_")[0]
            frame_id = id.split("_")[1]
            content_line = '{},{},{},{},{},{},{},{}'.format(int(video_id), int(frame_id), int(bbox[0]),
                                                            int(bbox[1]), int(bbox[2]) - int(bbox[0]),
                                                            int(bbox[3]) - int(bbox[1]), int(labels[idx]),
                                                            scores[idx])
            width = int(bbox[2]) - int(bbox[0])
            height = int(bbox[3]) - int(bbox[1])
            # if width < 40 or height < 40: continue
            # print(content_line)
            output_txt_file.write(content_line)
            output_txt_file.write('\n')

def fine_tune_emsemble(output_txt_file, image_ids):
    effdet_ed7_896_box_pred = load_dict('effdet_ed7_896_box_pred.pkl')
    effdet_ed7_896_score_pred = load_dict('effdet_ed7_896_score_pred.pkl')
    effdet_ed7_896_label_pred = load_dict('effdet_ed7_896_label_pred.pkl')

    for id, obj_list in effdet_ed7_896_box_pred.items():
        object_list = []
        remove_indexes = []
        for idx, objs in enumerate(obj_list):
            if len(objs) == 0: continue
            for i, object in enumerate(objs):
                boxes = np.array(object[:4])
                if float(effdet_ed7_896_score_pred[id][idx][i]) < 0.3: continue
                if len(boxes) > 0:
                    height, width = 1080, 1920
                    boxes[[0, 2]] = (boxes[[0, 2]] * width).clip(min=0, max=width - 1)
                    boxes[[1, 3]] = (boxes[[1, 3]] * height).clip(min=0, max=height - 1)
                    object_list.append([int(boxes[0]), int(boxes[1]), int(boxes[2]),
                                    int(boxes[3]), int(effdet_ed7_896_label_pred[id][idx][i]),
                                    effdet_ed7_896_score_pred[id][idx][i]])

        num_vehicle = len(object_list)
        object_list = np.array(object_list)
        video_id = id.split("_")[0]
        frame_id = id.split("_")[1]
        for i in range(0, num_vehicle - 1):
            for j in range(i + 1, num_vehicle):
                if overlap_ratio(object_list[i], object_list[j]) > 0.9:
                    if object_list[j][-1] > object_list[i][-1]:
                        remove_indexes.append(i)
                    else:
                        remove_indexes.append(j)
        new_objects = np.delete(object_list, remove_indexes, axis=0)
        for obj in new_objects:
            content_line = '{},{},{},{},{},{},{},{}'.format(video_id, frame_id, int(obj[0]),
                                                            int(obj[1]), int(obj[2]) - int(obj[0]),
                                                            int(obj[3]) - int(obj[1]), int(obj[4]),
                                                            obj[5])
            output_txt_file.write(content_line)
            output_txt_file.write('\n')



if __name__ == '__main__':
    original_label_path = 'effdet_ed6_640_fold1.txt'
    # original_label_path = 'effdet_ed7_896_fold1_finetune.txt'
    # original_label_path = 'label_new.txt'

    # overlap_thres = 0.95
    random.seed(42)

    df = 'dataset/test/'
    image_file_list = [f for f in listdir(df) if isfile(join(df, f))]
    image_ids = []
    for id in image_file_list:
        image_ids.append(id.split(".")[0])

    original_bbox_info = get_original_box(original_label_path)

    # video_file_path = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/videos_test/'
    video_file_path = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/videos/'

    video_file_path_out = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/videos_test_out_finetune/'
    video_file_list = [f for f in listdir(video_file_path) if isfile(join(video_file_path, f))]
    txt_path_test = 'effdet_ed6_640_only5.txt'
    txt_path = 'effdet_ed6_640_pseudo.txt'
    # output_txt_file = open(txt_path, "w")
    output_txt_file = open(txt_path_test, "w")

    # finetune_listvideo(video_file_list, original_bbox_info, output_txt_file, overlap_thres=0.9)
    # fine_tune_obj(original_bbox_info, output_txt_file, overlap_thres=0.9)
    emsemble_obj(output_txt_file, image_ids)
    # fine_tune_emsemble(output_txt_file, image_ids)

    # for video_id, frame_info in original_bbox_info.items():
    #     for frame_id, object_list in frame_info.items():
    #         image_id = "{}_{}".format(video_id, frame_id)
    #
    #         remove_indexes = []
    #
    #         num_vehicle = len(object_list)
    #         object_list = np.array(object_list)
    #
    #         boxes = object_list[:,:4]
    #         boxes[:, [0, 2]] = (boxes[:, [0, 2]] / 1920)
    #         boxes[:, [1, 3]] = (boxes[:, [1, 3]] / 1080)
    #         scores = object_list[:,5]
    #         labels = object_list[:,4]
    #
    #         boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=0.5
    #                                                       , skip_box_thr=0.44)
    #         idxs = np.where(scores > 0.35)[0]
    #         boxes = boxes[idxs]
    #         scores = scores[idxs]
    #         labels = labels[idxs]
    #
    #         if len(boxes) > 0:
    #             height, width = 1080, 1920
    #             boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(min=0, max=width - 1)
    #             boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(min=0, max=height - 1)
    #         for idx, box in enumerate(boxes):
    #             content_line = '{},{},{},{},{},{},{},{}'.format(video_id, frame_id, int(box[0]),
    #                                                             int(box[1]), int(box[2]) - int(box[0]),
    #                                                             int(box[3]) - int(box[1]), int(labels[idx]),
    #                                                             scores[idx])
    #             width = int(box[2]) - int(box[0])
    #             height = int(box[3]) - int(box[1])
    #             if width < 40 or height < 40: continue
    #             output_txt_file.write(content_line)
    #             output_txt_file.write('\n')


            # for i in range(0, num_vehicle - 1):
            #     for j in range(i + 1, num_vehicle):
            #         if overlap_ratio(object_list[i], object_list[j]) > 0.9:
            #             if object_list[j][-1] > object_list[i][-1]:
            #                 remove_indexes.append(i)
            #             else:
            #                 remove_indexes.append(j)
            #
            # new_objects = np.delete(object_list, remove_indexes, axis=0)
            # print("hehe")
            # for obj in new_objects:
            #     content_line = '{},{},{},{},{},{},{},{}'.format(video_id, frame_id, int(obj[0]),
            #                                                     int(obj[1]), int(obj[2]) - int(obj[0]),
            #                                                     int(obj[3]) - int(obj[1]), int(obj[4]),
            #                                                     obj[5])
            #     width = int(obj[2]) - int(obj[0])
            #     height = int(obj[3]) - int(obj[1])
            #     if width < 40 or height < 40: continue
            #     if float(obj[5]) < 0.35: continue
            #     output_txt_file.write(content_line)
            #     output_txt_file.write('\n')












