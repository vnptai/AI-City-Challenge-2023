import os.path
from glob import glob

import cv2
from post_processing_for_tracking.track_object import tracker
import argparse
import numpy as np
from object_association.object_association import Object_Association

CLASSES = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'crowd_human']

arg = argparse.ArgumentParser()
arg.add_argument('--video', required=False,
                help='video-name', default="/data/huytq/data/AIcitychallenge/track5/aicity2023_track5/aicity2023_track5_test/videos/")
args = arg.parse_args()
obj_association = Object_Association(prediction_path='detection_results/pseudo_p1_combination.txt',
                head_label_path='detection_results/effdet_ed7_768_head.txt',
                video_folder=args.video,
                write_to_video=False,
                display=False,
                remove_small_objs_rule=True,
                overlap_60_rule=True,
                remove_motor_without_driver=False,
                head_thresh=0.1,
                conf_thres=0.2, 
                head_motor_overlap_thresh=0.75)
# 0.2 - 334
# 0.1 - 390
conf_class_motor, conf_class_D, conf_class_D_No, conf_class_P1, cond_class_P1_No, conf_class_P2, conf_class_P2_No = 0.35,0.2,0.2,0.2,0.2,0.2,0.2

c_class_motor, c_class_D, c_class_D_No, c_class_P1, c_class_P1_No, c_class_P2, c_class_P2_No = 0, 0, 0, 0, 0, 0, 0
tracking = tracker.my_tracking(max_age=5, min_hits=0, iou_threshold=0.15, enable_voting=False)
file_output = open("results.txt", 'w')
output = []

for video_path in glob(os.path.join(args.video, '*.mp4')):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_id = 0
    c_frame_miss = 0
    video_id = int(video_path.split('/')[-1].split('.')[0])

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('{}.mp4'.format(video_name), fourcc, int(cap.get(5)),
    #                     (frame_width, frame_height))
    #
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break
            _, _, _ = frame.shape
            
        except Exception as e:
            print(e)
            break

        frame_id += 1
        results = obj_association.foward_frame(frame, video_id, frame_id)
        bbox_human = []
        for rs in results:
            heads = rs.heads
            box = rs.get_box_info()
            bbox_human.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
            # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            #               (255, 0, 0),
            #               2)
            # humans = rs.humans
            # for human in humans:
            #     box = human.get_box_info()
            #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            #                   (255, 0, 0),
            #                   2)
            # #     heads = human.heads
            for head in heads:
                box = head.get_box_info()
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            (255, 0, 0),
                            2)
                cv2.putText(frame, str(head.motor_id[:8]), (int(box[0]) - 5, int(box[1])), color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)

        # output_tracker = tracking.update(np.array(results))
        # print("len {} frame {}".format(len(results),frame_id))
        output_tracker = []

        if len(bbox_human) > 0:
            output_motor, output_human = tracking.update(np.array(bbox_human), results)
            # print("len tracker {} frame {}".format(len(output_tracker), frame_id))
            for box in output_motor:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            (255, 0, 255),
                            2)
                x, y = box[0], box[1]
                direction = ""
                if box[7] != 2:
                    direction = "IN" if box[7] == 1 else "OUT"
                text = "ID {} - {}-{}-".format(int(box[6]), int(box[5]), round(box[4], 3)) + direction
                cv2.putText(frame, text,
                            (int(x), int(y - 8)),
                            cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                output_tracker.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4]), int(box[5])])
            for box in output_human:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            (0, 125, 255),
                            2)
                x, y = box[0], box[1]
                text = "{}-{}".format(int(box[4]), round(box[5], 3))
                cv2.putText(frame, text,
                            (int(x), int(y - 8)),
                            cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                output_tracker.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[5]), int(box[4])])

        output_bbox = output_tracker
        for box in output_bbox:
            conf = box[4]
            classid = int(box[5])
            if classid == 0:
                if conf < conf_class_motor:
                    continue
                c_class_motor += 1
            elif classid == 1:
                if conf < conf_class_D:
                    continue
                c_class_D += 1
            elif classid == 2:
                if conf < conf_class_D_No:
                    continue
                c_class_D_No += 1
            elif classid == 3:
                if conf < conf_class_P1:
                    continue
                c_class_P1 += 1
            elif classid == 4:
                if conf < cond_class_P1_No:
                    continue
                c_class_P1_No += 1
            elif classid == 5:
                if conf < conf_class_P2:
                    continue
                c_class_P2 += 1
            elif classid == 6:
                if conf < conf_class_P2_No:
                    continue
                c_class_P2_No += 1
            xmin = 1 if box[0] < 1 else box[0]
            ymin = 1 if box[1] < 1 else box[1]
            xmax = 1920 if box[2] > 1920 else box[2]
            ymax = 1080 if box[3] > 1080 else box[3]
            # if int(box[5]) == 6:
            output.append("%d,%d,%d,%d,%d,%d,%d,%.6f" % (
                int(video_name), int(frame_id), xmin, ymin, xmax - xmin, ymax - ymin,
                int(box[5]) + 1,
                float(box[4])))  # tracking
        # out.write(frame)
        # cv2.imshow("image", cv2.resize(frame, dsize=(960, 640)))
        # k = cv2.waitKey(0)
        # if k == 27:
        #     break


def sorter(item):
    item = item.split(",")
    return int(item[0])


output = sorted(output, key=sorter)
file_output.write("\n".join(output))
