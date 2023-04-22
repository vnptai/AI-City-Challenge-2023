import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from itertools import product
import torch
from torch.utils.data import DataLoader
has_apex = False
from models import fasterrcnn_resnet_fpn, get_effdet_test
from dataset import TTAHorizontalFlip, TTAVerticalFlip, TTARotate90, TTACompose, AICityTestDataset
from utils import save_dict, collate_fn
from os import listdir
from os.path import isfile, join
import warnings
import cv2

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--network", default="effdet", type=str, choices=['effdet'])
parser.add_argument("--backbone", default="ed6", type=str,
                    choices=['ed0', 'ed1', 'ed2', 'ed3', 'ed4', 'ed5', 'ed6', 'ed7'])
parser.add_argument("--img-size", default=640, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--workers", default=16, type=int)
parser.add_argument("--test-dir", default='../aicity_dataset/aicity2023_track5_test_images/', type=str)
parser.add_argument("--checkpoint-dir", default='checkpoints', type=str)
parser.add_argument("--folds", nargs="+", type=int)
# parser.add_argument("--folds", default=[1])
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    # output_txt_file = open(args.test_csv, "w")
    torch.cuda.set_device('cuda:0')

    test_dataset = AICityTestDataset(df=args.test_dir, img_size=args.img_size, root_dir=args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             collate_fn=collate_fn)

    tta_transforms = []
    for tta_combination in product([TTAHorizontalFlip(args.img_size), None], [TTAVerticalFlip(args.img_size), None], [TTARotate90(args.img_size), None]):
        tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))




    box_pred = {}
    score_pred = {}
    label_pred = {}
    for image_id in test_dataset.image_ids:
        box_pred[image_id] = []
        score_pred[image_id] = []
        label_pred[image_id] = []

    for fold in args.folds:
        model = get_effdet_test(backbone=args.backbone, num_classes=7, img_size=args.img_size)
        model = model.cuda()

        checkpoint = torch.load(
            '{}/{}_{}_{}_fold{}.pth'.format(args.checkpoint_dir, args.network, args.backbone, args.img_size, fold),
        map_location='cuda:0')
        # checkpoint = torch.load(
        #     '{}/{}_{}_{}_fold{}.pth'.format(args.checkpoint_dir, args.network, args.backbone, args.img_size, fold))
        model.model.load_state_dict(checkpoint)
        del checkpoint
        gc.collect()

        model.eval()

        for images, image_ids in tqdm(test_loader):
            images = torch.stack(images)
            images = images.cuda()
            for tta_transform in tta_transforms:
                with torch.set_grad_enabled(False):
                    dets = model(tta_transform.effdet_augment(images.clone()),
                                 torch.tensor([1] * images.shape[0]).float().cuda())

                for det, image_id in zip(dets, image_ids):
                    boxes = det.detach().cpu().numpy()[:, :4]
                    scores = det.detach().cpu().numpy()[:, 4]
                    labels = det.detach().cpu().numpy()[:, 5]
                    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                    boxes = tta_transform.deaugment_boxes(boxes)

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

        del model
    del tta_transforms
    torch.cuda.empty_cache()
    gc.collect()

    save_dict(box_pred, 'pkl/{}_{}_{}_box_pred.pkl'.format(args.network, args.backbone, args.img_size))
    save_dict(score_pred, 'pkl/{}_{}_{}_score_pred.pkl'.format(args.network, args.backbone, args.img_size))
    save_dict(label_pred, 'pkl/{}_{}_{}_label_pred.pkl'.format(args.network, args.backbone, args.img_size))
