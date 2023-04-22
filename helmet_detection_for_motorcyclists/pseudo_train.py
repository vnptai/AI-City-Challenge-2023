import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import math
import gc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

has_apex = False

from models import get_effdet_train
from dataset import AICityTrainDataset
from utils import collate_fn

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backbone", default="ed6", type=str,
                    choices=['ed0', 'ed1', 'ed2', 'ed3', 'ed4', 'ed5', 'ed6', 'ed7'])
parser.add_argument("--img-size", default=640, type=int)
parser.add_argument("--batch-size", default=1, type=int)
parser.add_argument("--workers", default=16, type=int)
parser.add_argument("--train-csv", default='train.csv', type=str)
parser.add_argument("--valid-csv", default='valid.csv', type=str)
parser.add_argument("--pretrain-path", default=None, type=str)
parser.add_argument("--checkpoint-path", default=None, type=str)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--init-lr", default=8e-5, type=float)
parser.add_argument("--mixup", default=True, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--use-amp", default=True, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--load-optimizer", default=False, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--save-optimizer", default=False, type=lambda x: (str(x).lower() == "true"))
args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(train_loader, valid_loader, ckpt_path, pretrain_path, epochs=10, init_lr=1e-4):
    model = get_effdet_train(args.backbone, num_classes=7, img_size=args.img_size)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    if args.load_optimizer:
        ckpt = torch.load(pretrain_path, map_location='cuda:0')
        model.model.load_state_dict(ckpt['model'])
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt['val_loss_min'] is not None:
            val_loss_min = ckpt['val_loss_min']
        else:
            val_loss_min = np.Inf
        del ckpt
        gc.collect()
    else:
        model.model.load_state_dict(torch.load(pretrain_path, map_location='cuda:0'))
        val_loss_min = np.Inf

    lf = lambda x: (((1 + math.cos(x * math.pi / args.epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = 0

    loss_hist = AverageMeter()
    for epoch in range(epochs):
        loss_hist.reset()
        model.train()

        loop = tqdm(enumerate(train_loader))
        for i, (images, targets) in loop:
            ### mixup
            if random.random() > 0.5 and args.mixup:
                images = torch.stack(images).cuda()
                shuffle_indices = torch.randperm(images.size(0))
                indices = torch.arange(images.size(0))
                lam = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)
                images = lam * images + (1 - lam) * images[shuffle_indices, :]
                mix_targets = []
                for i, si in zip(indices, shuffle_indices):
                    if i.item() == si.item():
                        target = targets[i.item()]
                    else:
                        target = {
                            'boxes': torch.cat([targets[i.item()]['boxes'], targets[si.item()]['boxes']]),
                            'labels': torch.cat([targets[i.item()]['labels'], targets[si.item()]['labels']])
                        }

                    mix_targets.append(target)
                targets = mix_targets
            else:
                images = torch.stack(images).cuda()
            boxes = [target['boxes'].cuda().float() for target in targets]
            labels = [target['labels'].cuda().float() for target in targets]

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                loss, _, _ = model(images, boxes, labels)
                if loss == 0 or not torch.isfinite(loss):
                    continue
                loss.backward()
                loss_hist.update(loss.detach().item(), images.size(0))
                optimizer.step()
            loop.set_description(
                'Epoch {:03d}/{:03d} | LR: {:.5f}'.format(epoch, epochs - 1, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=loss_hist.avg)
        train_loss = loss_hist.avg

        scheduler.step()

        model.eval()
        loss_hist.reset()
        for images, targets in tqdm(valid_loader):
            images = torch.stack(images).cuda()
            boxes = [target['boxes'].cuda().float() for target in targets]
            labels = [target['labels'].cuda().float() for target in targets]

            with torch.set_grad_enabled(False):
                loss, _, _ = model(images, boxes, labels)
                loss_hist.update(loss.detach().item(), images.size(0))
        val_loss = loss_hist.avg

        print('Train loss: {:.5f} | Val loss: {:.5f}'.format(train_loss, val_loss))

        if val_loss < val_loss_min:
            print('Valid loss improved from {:.5f} to {:.5f} saving model to {}'.format(val_loss_min, val_loss,
                                                                                        ckpt_path))
            val_loss_min = val_loss
            if args.save_optimizer:
                torch.save({
                    'model': model.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss_min': val_loss_min
                }, ckpt_path)
            else:
                torch.save(model.model.state_dict(), ckpt_path)
        if epoch == epochs - 1:
            break


if __name__ == "__main__":
    torch.cuda.set_device('cuda:0')
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)

    train_dataset = AICityTrainDataset(df=train_df, img_size=args.img_size, mode='train')
    valid_dataset = AICityTrainDataset(df=valid_df, img_size=args.img_size, mode='valid')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))

    train_model(train_loader,
                valid_loader,
                ckpt_path=args.checkpoint_path,
                pretrain_path=args.pretrain_path,
                epochs=args.epochs,
                init_lr=args.init_lr)