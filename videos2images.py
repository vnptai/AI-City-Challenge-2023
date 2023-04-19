import os
import cv2
import argparse
from glob import glob
import numpy as np


class Video_Splitter(object):
    def __init__(self,video_path, save_folder, save_image_path):
        """Video visualizer initialize
        Args:
            video_path (string): path to video
            save_folder (string): path to save visualized video
        """
        self.video_path = video_path
        self.video_id = int(video_path.split('/')[-1].split('.')[0])
        self.video_cap = cv2.VideoCapture(video_path)
        self.video_width  = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.video_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.save_image_path = save_image_path
    
    def export(self):
        if self.video_cap.isOpened() == False:
            print('Error openning video stream or file')
            return -1
        index = 1
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret == True:
                cv2.imwrite(os.path.join(self.save_image_path, str(index) + '.jpg'), frame)
                index += 1
            else:
                break
        
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', default='./aicity_dataset/aicity2023_track5/aicity2023_track5/videos', \
                        required=False ,help='path to aicity challenge videos folder')
    parser.add_argument('--save_folder', default='./aicity_dataset/aicity2023_track5_images', \
                        required=False ,help='path to save images')
    # parser.add_argument()
    args = parser.parse_args()
    return args
    
    
if __name__ == '__main__':
    args = parse_opt()
    os.makedirs(args.save_folder, exist_ok=True)
    for video_path in glob(os.path.join(args.video_folder, '*mp4')):
        video_id = video_path.split('/')[-1].split('.')[0]
        save_image_path = os.path.join(args.save_folder, 'images', video_id)
        os.makedirs(save_image_path, exist_ok=True)
        video_splitter = Video_Splitter(video_path, args.save_folder, save_image_path)
        video_splitter.export()
