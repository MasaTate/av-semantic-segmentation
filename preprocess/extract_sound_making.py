from configparser import Interpolation
import cv2
import numpy as np
import argparse
import glob
from tqdm import tqdm
import os
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=str, default=".", help="dataset root directory")

args = parser.parse_args()

colors = [[142,0,0], [100,80,0], [230,0,0]] #car, train, motorcycle

"""
test = np.array([[[0,0,142], [0,80,100], [0,0,230]]])
print(test.shape)
print(np.where((colors == test[0][1]).all(axis=1)))
"""

def mask_color(bg):
    bg = np.array(bg, dtype=np.uint8)
    mask = np.full((bg.shape[0], bg.shape[1]), 0, dtype=np.uint8)
    #print(bg.shape)
    for i in range(bg.shape[0]):
        for j in range(bg.shape[1]):
            index = np.where((colors == bg[i][j]).all(axis=1))[0]
            if len(index) != 0:
                if index[0] == 0:
                    mask[i][j] = 13
                elif index[0] == 1:
                    mask[i][j] = 16
                elif index[0] == 2:
                    mask[i][j] = 17
    return mask


for sc in tqdm(range(1, 166)):
    fdir = args.root + "/scene%04d/"%sc
    videonum = int(glob.glob(fdir+"/*_bg.png")[0].split('/')[-1].split('_')[1])
    bg_file = "VIDEO_"+"%04d"%videonum+"_bg_seman.png"
    savedir = fdir+"binary_mask_full"

    #print(fdir+bg_file)
    bg = cv2.imread(fdir+bg_file)

    height, width, _ = bg.shape
    bg = cv2.resize(bg, (int(width*0.5), int(height*0.5)))
    
    bg_mask = mask_color(bg)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print("bg shape : ", bg.shape)
    pred_path = sorted(glob.glob(fdir+"gtDLab/*_labelIds.png"))
    for img_path in tqdm(pred_path):
        label = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        label_mask = np.full(label.shape, 0, dtype=np.uint8)

        #car
        car_label = label.copy()
        car_label[bg_mask==13] = 0
        label_mask[car_label==26] = 255
        #train
        train_label = label.copy()
        train_label[bg_mask==16] = 0
        label_mask[train_label==31] = 255
        #motorcycle
        motor_label = label.copy()
        motor_label[bg_mask==17] = 0
        label_mask[motor_label==32] = 255

        mask = cv2.cvtColor(label_mask, cv2.COLOR_GRAY2BGR)
        prefix = img_path.split("/")[-1].split('.')[0].split('_')[0:4]
        save_name = "_".join(prefix) + "_mask.png"
        cv2.imwrite(os.path.join(savedir, save_name), mask)        
    