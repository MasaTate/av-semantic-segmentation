import cv2
import numpy as np
import argparse
import glob
import scipy.stats as stats
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=str, default=".", help="dataset root directory")

args = parser.parse_args()


for sc in tqdm(range(1, 166)):
    subdir = "/dataset_public/scene%04d/"%sc
    fdir = args.root + subdir
    videonum = int(glob.glob(fdir+"/*_cropped.mp4")[0].split('/')[-1].split('_')[1])
    video_file = "VIDEO_"+"%04d"%videonum+"_cropped.mp4"
    
    cap = cv2.VideoCapture(fdir+video_file)
    totalframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    out_frame = np.zeros((height, width, 3))
    
    w_patch = 3840
    h_patch = 1920
    frame_speed_rate = 3
    for w in range(0, width//w_patch):
        for h in range(0, height//h_patch):
            tmp_frame = np.zeros((h_patch, w_patch, 3, totalframe // frame_speed_rate))
            for i in tqdm(range(0, totalframe // frame_speed_rate)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_speed_rate)
                ret, frame = cap.read()
                tmp_frame[:,:,:,i] = frame[h*h_patch:(h+1)*h_patch, w*w_patch:(w+1)*w_patch, :]
            #mode_val, _ = stats.mode(tmp_frame, axis=3)
            tmp_frame = torch.from_numpy(tmp_frame)
            mode_val, _ = torch.mode(tmp_frame, dim=-1)
            #print(mode_val)
            mode_val = mode_val.detach().numpy()
            #print(mode_val.shape)
            out_frame[h*h_patch:(h+1)*h_patch, w*w_patch:(w+1)*w_patch, :] = mode_val
            #print(out_frame.shape)
    
    cv2.imwrite(fdir+"VIDEO_"+"%04d"%videonum+"_bg.png", out_frame)