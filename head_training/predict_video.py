import os

import numpy as np
import pandas as pd
import argparse
from albumentations import *
import gc
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import timm
import pickle
import multiprocessing
from multiprocessing import Pool
import random
from os import listdir
from os.path import isfile, join

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from apex import amp
from evaluation import calculate_final_score
from dataset import WheatTestset
from dataset import TTAHorizontalFlip, TTAVerticalFlip, TTARotate90, TTACompose
from itertools import product

from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet.efficientdet import HeadNet
from models import fasterrcnn_resnet_fpn, get_effdet
from utils import save_dict, MyThresh, wbf_optimize

from PIL import Image
from matplotlib import pyplot as plt
import cv2

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--network", default="effdet", type=str, choices=['fasterrcnn', 'effdet'])
parser.add_argument("--backbone", default="ed7", type=str,
                    choices=['ed0', 'ed1', 'ed2', 'ed3', 'ed4', 'ed5', 'ed6', 'ed7', 'resnet50', 'resnet101',
                             'resnet152'])
parser.add_argument("--img-size", default=768, type=int)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--folds", default=[1], type=int)
# parser.add_argument("--folds", nargs="+", type=int)
parser.add_argument("--use-amp", default=False, type=lambda x: (str(x).lower() == "true"))
args = parser.parse_args()

if args.network == 'fasterrcnn':
    args.use_amp = False
else:
    args.use_amp = False

import warnings

warnings.filterwarnings("ignore")

def convert_image(image, img_size, transforms):
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    img = np.array(pil_image)
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = transforms(image=img)['image']
    img = img.astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)

    return img

def collate_fn(batch):
    return tuple(zip(*batch))

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
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    # df = pd.read_csv('dataset/trainset_ai.csv')

    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(7)]
    pp_threshold = 0.22
    nms_threshold = 0.32
    box_threshold = 0.2

    ground_truth = {}
    box_pred = {}
    score_pred = {}
    label_pred = {}

    transforms = Resize(height=args.img_size, width=args.img_size, interpolation=1, p=1)

    if args.network == 'fasterrcnn':
        model = fasterrcnn_resnet_fpn(backbone_name=args.backbone, pretrained=False, pretrained_backbone=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model = model.cuda()
    elif args.network == 'effdet':
        model = get_effdet(args.backbone, num_classes=1, img_size=args.img_size, mode='valid', pretrained=False,
                           pretrained_backbone=False)
        model = model.cuda()
        if args.use_amp:
            model = amp.initialize(model, opt_level='O1')
    else:
        raise ValueError('NETWORK')

    CHECKPOINT = '{}_{}_{}_fold{}.pth'.format(args.network, args.backbone, args.img_size, args.folds[0])
    checkpoint = torch.load(CHECKPOINT, map_location='cuda:0')
    if args.network == 'effdet':
        model.model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    del checkpoint
    gc.collect()

    model.eval()

    idx_test = 0

    video_file_path = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/videos_test/'
    video_file_path_out = '/media/hungdv/Source/Data/ai-city-challenge/aicity2023_track5/videos_test_out/'
    video_file_list = [f for f in listdir(video_file_path) if isfile(join(video_file_path, f))]
    txt_path = '{}_{}_{}_fold{}.txt'.format(args.network, args.backbone, args.img_size, args.folds[0])
    output_txt_file = open(txt_path, "w")
    for video_file in video_file_list:
        video_path = video_file_path + video_file
        video_number = int(video_file.split(".")[0])
        if video_number != 42: continue
        cap = cv2.VideoCapture(video_path)
        index_frame = 0

        size = (1920, 1080)
        writer = cv2.VideoWriter(video_file_path_out + video_file,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret != True: break
            if args.network == 'fasterrcnn':
                images = list(image.cuda() for image in images)
                for tta_transform in tta_transforms:
                    with torch.set_grad_enabled(False):
                        outputs = model(tta_transform.fasterrcnn_augment(images.copy()))
                    for image_id, o in zip(image_ids, outputs):
                        boxes = o['boxes'].data.cpu().numpy()
                        scores = o['scores'].data.cpu().numpy()
                        labels = o['labels'].data.cpu().numpy()

                        boxes = tta_transform.deaugment_boxes(boxes)

                        scores *= 0.8  # normalize to efficientdet score

                        idxs = np.where(scores > 0.2)[0]
                        boxes = boxes[idxs]
                        scores = scores[idxs]
                        labels = labels[idxs]

                        if boxes.shape[0] > 0:
                            boxes /= float(args.img_size)
                            boxes = boxes.clip(min=0, max=1)

                        box_pred[image_id].append(boxes.tolist())
                        score_pred[image_id].append(scores.tolist())
                        label_pred[image_id].append(labels.tolist())

            else:
                index_frame += 1
                if index_frame != 40: continue
                images = convert_image(frame, args.img_size, transforms)

                images = torch.stack([images])
                images = images.cuda()


                with torch.set_grad_enabled(False):
                    dets = model(images.clone(),
                                 torch.tensor([1] * images.shape[0]).cuda())
                for det in dets:
                    boxes = det.detach().cpu().numpy()[:, :4]
                    scores = det.detach().cpu().numpy()[:, 4]
                    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                    labels = det.detach().cpu().numpy()[:, 5]
                    # if (max(labels) > 6): print(max(labels))
                    idxs = np.where(scores > 0.2)[0]
                    boxes = boxes[idxs]
                    scores = scores[idxs]
                    labels = labels[idxs]

                    if boxes.shape[0] > 0:
                        boxes /= float(args.img_size)
                        boxes = boxes.clip(min=0, max=1)

                    boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None,
                                                                  iou_thr=nms_threshold, skip_box_thr=box_threshold)

                    boxes[:, 0] = boxes[:, 0] * 1920
                    boxes[:, 1] = boxes[:, 1] * 1080
                    boxes[:, 2] = boxes[:, 2] * 1920
                    boxes[:, 3] = boxes[:, 3] * 1080

                    boxes = boxes.astype(int)
                    # image_path = "/media/hungdv/Source/Code/AIChallenge/global-wheat-dection-2020/dataset/train/" + \
                    #              str(image_id)+ ".jpg"
                    # image_show = cv2.imread(image_path)
                    for idx, box in enumerate(boxes):
                        content_line = '{},{},{},{},{},{},{},{}'.format(video_number,index_frame,box[0],
                                                                      box[1],box[2]-box[0],
                                                                      box[3]-box[1], int(labels[idx]),
                                                                        scores[idx])
                        output_txt_file.write(content_line)
                        output_txt_file.write('\n')
                        plot_one_box([box[0], box[1], box[2], box[3]], frame, color=colors[int(labels[idx])],
                                     label=str(labels[idx]), line_thickness=True)
                    cv2.imwrite("2.jpg", frame)
                    # writer.write(frame)
                    # cv2.imshow('image', frame)
                    # if cv2.waitKey(50) & 0xFF == ord('q'):
                    #     break

        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("The video " + video_file_path_out + video_file + " was successfully saved")
    del model
    output_txt_file.close()
    gc.collect()
