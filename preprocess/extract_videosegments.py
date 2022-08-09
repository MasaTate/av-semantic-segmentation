import glob,os
import numpy as np
import argparse
from tqdm import tqdm


import cv2
import ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir",default=".",help="dataset input root directory")
parser.add_argument("--out_dir",help="dataset output root directory")
args = parser.parse_args()

for sc in tqdm(range(1, 166)):
    in_fdir = args.in_dir + "/scene%04d/"%sc
    out_fdir = args.out_dir + "/scene%04d/"%sc
    videonum = int(glob.glob(in_fdir+"/*cropped.mp4")[0].split('/')[-1].split('_')[1]);
    videofile="VIDEO_"+"%04d"%videonum+"_cropped.mp4"
    save_dir = out_fdir + "split_videoframes/"
    
    video_duration_float = float(ffmpeg.probe(in_fdir + videofile)['streams'][0]['duration'])
    video_duration = np.floor(video_duration_float)
    audio_duration_float = float(ffmpeg.probe(in_fdir + videofile)['streams'][1]['duration'])
    audio_duration = np.floor(audio_duration_float)
    diff = audio_duration - video_duration
    print("video duration : ", video_duration, "(", video_duration_float, ")")
    print("audio duration : ", audio_duration, "(", audio_duration_float, ")")
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for t in tqdm(np.arange(1, audio_duration, 1)):
        save_path = save_dir+videofile[:-4]+"_%06d"%(t-1)+".png"
        ffmpeg.input(in_fdir + videofile, ss=t).filter("scale", 1920, -1).output(save_path, vframes=1).run(quiet=True, overwrite_output=True)  
