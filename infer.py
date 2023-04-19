import os.path
import cv2
from post_processing_for_tracking.track_object import tracker
import argparse
import numpy as np
import tqdm
import glob
from object_association.object_association import Object_Association

CLASSES = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet']

arg = argparse.ArgumentParser()
arg.add_argument('--video-folder', required=False,
                help='video-folder', default="/data/AIcitychallenge/track5/aicity2023_track5/aicity2023_track5_test/videos/")
args = arg.parse_args()

obj_association = Object_Association(video_folder=args.video_folder,
            display=False,
            prediction_path='./baseline_training/pseudo.txt',
            head_label_path='./head_training/effdet_ed7_head.txt')

conf_class_motor, conf_class_D, conf_class_D_No, conf_class_P1, cond_class_P1_No, conf_class_P2, conf_class_P2_No = 0.35, 0.32, 0.32, 0.32, 0.32, 0.2, 0.2
file_output = open("final_results.txt", "w")

def sorter(item):
    item = item.split(",")
    return int(item[1])


for video_path in tqdm.tqdm(sorted(glob.glob(args.video_folder + "/*"))):
    cap = cv2.VideoCapture(video_path)
    tracking = tracker.my_tracking()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_id = 0
    output = []
    bbox_value_old = []
    bbox_id_old = []
    c_frame_miss = 0
    video_id = int(video_path.split('/')[-1].split('.')[0])

    while cap.isOpened():
        try:
            _, frame = cap.read()
            _, _, _ = frame.shape
        except Exception as e:
            break
        frame_id += 1
        results_obj_association = obj_association.foward_frame(frame, video_id, frame_id)
        bbox_motor = []
        for rs in results_obj_association:
            box = rs.get_box_info()
            bbox_motor.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
        output_tracker = []
        if len(bbox_motor) > 0:
            output_vehicle, output_human = tracking.update(np.array(bbox_motor), results_obj_association)
            for box in output_vehicle:
                output_tracker.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]), int(box[5])])
            for box in output_human:
                output_tracker.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[5]), int(box[4])])
        output_bbox = np.array(output_tracker)
        for box in output_bbox:
            conf = box[4]
            classid = int(box[5])
            if classid == 0:
                if conf < conf_class_motor:
                    continue
            elif classid == 1:
                if conf < conf_class_D:
                    continue
            elif classid == 2:
                if conf < conf_class_D_No:
                    continue
            elif classid == 3:
                if conf < conf_class_P1:
                    continue
            elif classid == 4:
                if conf < cond_class_P1_No:
                    continue
            elif classid == 5:
                if conf < conf_class_P2:
                    continue
            elif classid == 6:
                if conf < conf_class_P2_No:
                    continue
            xmin = 1 if box[0] < 1 else box[0]
            ymin = 1 if box[1] < 1 else box[1]
            xmax = 1920 if box[2] > 1920 else box[2]
            ymax = 1080 if box[3] > 1080 else box[3]
            output.append("%d,%d,%d,%d,%d,%d,%d,%.6f" % (
                int(video_name), int(frame_id), xmin, ymin, xmax-xmin, ymax-ymin,
                int(box[5]) + 1,
                float(box[4])))
    output = sorted(output, key=sorter)
    file_output.write("\n".join(output))
    if len(output) > 0:
        file_output.write("\n")

