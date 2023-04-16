import csv
from os import listdir
from os.path import isfile, join
import cv2
import os
import numpy as np
import csv

def xywhn2xyxy(x, w=1920, h=1080, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def convert_head_csv():
    header = ['image_id', 'fold', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox', 'source']
    output_file = "dataset/trainset_head.csv"
    label_txt = "label_head.txt"
    label_writer = open(label_txt, "w")

    with open(output_file, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(header)

        folder_dir = '/media/hungdv/Source/Data/ai-city-challenge/head_body_labels/'
        folder_list = [name for name in os.listdir(folder_dir) if os.path.isdir(os.path.join(folder_dir, name))]
        for label_folder in folder_list:
            folder_path = folder_dir + label_folder
            label_file_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
            for file_path in label_file_list:
                if file_path in ['classes.txt', 'static_objects.txt']:continue
                f = open(folder_path + "/" + file_path)
                # print(folder_path)
                for line in f.readlines():
                    data = line.split(' ')
                    video_id = int(label_folder)
                    frame = int(file_path.split(".")[0])
                    bb_left_center, bb_top_center, bb_width, bb_height, class_id = data[1], data[2], data[3], data[4].split('\n')[0], data[0]
                    image_path = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/train/" + \
                                 str(video_id) + "_" + str(frame) + ".jpg"

                    if os.path.isfile(image_path) == False:
                        # print(image_path)
                        continue
                    xyxy = xywhn2xyxy(np.array([[float(bb_left_center), float(bb_top_center),
                                                 float(bb_width), float(bb_height)]]))[0]
                    class_id = int(class_id)
                    fold = int(video_id) % 5
                    x_min = float(xyxy[0])
                    y_min = float(xyxy[1])
                    x_max = float(xyxy[2])
                    y_max = float(xyxy[3])
                    if x_min > x_max or y_min > y_max:
                        print(xyxy)
                    if x_max > 1920.0: x_max = 1920.0

                    if y_max > 1080.0: y_max = 1080.0
                    image_id = str(video_id) + "_" + str(frame)
                    x_min, y_min, x_max, y_max = float(int(x_min)), float(int(y_min)), float(int(x_max)), float(int(y_max))
                    if int(class_id) == 1: continue
                    # print([image_id, fold, x_min, y_min, x_max, y_max, "True", str(class_id), class_id])
                    content_line = '{},{},{},{},{},{},{},{}'.format(int(video_id), int(frame), int(x_min),
                                                                    int(y_min), int(x_max) - int(x_min),
                                                                    int(y_max) - int(y_min), int(class_id),
                                                                    1)
                    label_writer.write(content_line)
                    label_writer.write('\n')
                    if int(x_max) - int(x_min) < 10 or int(y_max) - int(y_min) < 10:
                        print(xyxy)
                        continue
                    writer.writerow([image_id, fold, x_min, y_min, x_max, y_max, "True", class_id])


def convert_csv_new():
    header = ['image_id', 'fold', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox', 'source', 'label']
    output_file = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/trainset_ai.csv"
    label_txt = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/label_new.txt"
    label_writer = open(label_txt, "w")
    with open(output_file, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(header)

        folder_dir = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/aicity_labels/'
        folder_list = [name for name in os.listdir(folder_dir) if os.path.isdir(os.path.join(folder_dir, name))]
        for label_folder in folder_list:
            folder_path = folder_dir + label_folder
            label_file_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
            for file_path in label_file_list:
                if file_path in ['classes.txt', 'static_objects.txt']:continue
                f = open(folder_path + "/" + file_path)
                print(folder_path)
                for line in f.readlines():
                    data = line.split(' ')
                    video_id = int(label_folder)
                    frame = int(file_path.split(".")[0])
                    bb_left_center, bb_top_center, bb_width, bb_height, class_id = data[1], data[2], data[3], data[4].split('\n')[0], data[0]
                    image_path = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/train/" + \
                                 str(video_id) + "_" + str(frame) + ".jpg"

                    if os.path.isfile(image_path) == False:
                        # print(image_path)
                        continue
                    xyxy = xywhn2xyxy(np.array([[float(bb_left_center), float(bb_top_center),
                                                 float(bb_width), float(bb_height)]]))[0]
                    class_id = int(class_id) + 1
                    fold = int(video_id) % 5
                    x_min = float(xyxy[0])
                    y_min = float(xyxy[1])
                    x_max = float(xyxy[2])
                    if x_max > 1920.0: x_max = 1920.0
                    y_max = float(xyxy[3])
                    if y_max > 1080.0: y_max = 1080.0
                    image_id = str(video_id) + "_" + str(frame)
                    x_min, y_min, x_max, y_max = float(int(x_min)), float(int(y_min)), float(int(x_max)), float(int(y_max))
                    print([image_id, fold, x_min, y_min, x_max, y_max, "True", str(class_id), class_id])
                    content_line = '{},{},{},{},{},{},{},{}'.format(int(video_id), int(frame), int(x_min),
                                                                    int(y_min), int(x_max) - int(x_min),
                                                                    int(y_max) - int(y_min), int(class_id),
                                                                    1)
                    label_writer.write(content_line)
                    label_writer.write('\n')
                    writer.writerow([image_id, fold, x_min, y_min, x_max, y_max, "True", str(class_id), class_id])

def convert_csv():
    header = ['image_id','fold','xmin','ymin','xmax','ymax','isbox','source', 'label']
    output_file = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/trainset_ai_old.csv"
    with open(output_file, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(header)
        f = open("/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/gt.txt", "r")
        for line in f.readlines():
            data = line.split(',')
            video_id, frame, track_id, bb_left, bb_top, bb_width, bb_height, class_id = data[0], data[1],data[2],\
                data[3],data[4],data[5],data[6],data[7].split('\n')[0]

            image_path = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/train/" + \
                         str(video_id) + "_" + str(frame) + ".jpg"

            if os.path.isfile(image_path) == False:
                print(image_path)
                continue

            class_id = int(class_id)
            # if class_id == 2 or class_id == 3: class_id = 2
            # if class_id == 4 or class_id == 5: class_id = 3
            # if class_id == 6 or class_id == 7: class_id = 3
            # if class_id == 6: class_id = 4
            # if class_id == 7: class_id = 5
            fold = int(video_id) % 5
            x_min = float(bb_left)
            y_min = float(bb_top)
            x_max = float(bb_left) + float(bb_width)
            if x_max > 1920.0: x_max = 1920.0
            y_max = float(bb_top) + float(bb_height)
            if y_max > 1080.0: y_max = 1080.0
            image_id = video_id + "_" + frame
            writer.writerow([image_id, fold, x_min, y_min, x_max, y_max, "True", str(class_id), class_id])
            # print(line)

def gen_image():
    video_file_path = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/videos_test/'
    video_file_list = [f for f in listdir(video_file_path) if isfile(join(video_file_path, f))]
    output_file_path = '/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/test/'
    for video_file in video_file_list:
        video_path = video_file_path + video_file
        video_number = int(video_file.split(".")[0])
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            idx += 1
            frame_name = str(video_number) + "_" + str(idx)
            if ret == True:
                print(output_file_path + frame_name + ".jpg")
                image_name = output_file_path + frame_name + ".jpg"
                cv2.imwrite(image_name, frame)
            else:
                break
        cap.release()

def visualize_box(video_id_source, image_id_source):
    image_path = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/test/" + \
                 str(video_id_source) + "_" + str(image_id_source) + ".jpg"
    image = cv2.imread(image_path)
    f = open("best.txt", "r")
    bbox = []
    for line in f.readlines():
        data = line.split(',')
        video_id, frame, track_id, bb_left, bb_top, bb_width, bb_height, class_id = data[0], data[1],data[2],\
            data[3],data[4],data[5],data[6],data[7].split('\n')[0]
        if int(video_id) == video_id_source and int(frame) == image_id_source:
            bbox.append((int(bb_left), int(bb_top), int(bb_width), int(bb_height)))

    for box in bbox:
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    # convert_head_csv()
    visualize_box(20,20)
#
# print(video_file_list)