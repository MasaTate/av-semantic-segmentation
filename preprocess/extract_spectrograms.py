"""
Author: Arun Balajee Vasudevan
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import pyaudio
import wave
import sys,glob,os,librosa
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir",default=".",help="dataset input root directory")
parser.add_argument("--out_dir",help="dataset output root directory")
parser.add_argument("--tracks", default=[7, 8], help="input track list")
args = parser.parse_args()

# length of data to read.
chunk = 1024
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
import wavio

audio_length=2;

def desrired_rms(samples):
  rms = np.sqrt(np.mean(samples**2))
  return rms

def normalize(samples, desired_rms = 0.1, eps = 1e-4, rms=0):
  if rms==0: rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  else:    rms=rms
  samples = samples * (desired_rms / rms)
  return samples


def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def get_signal(wavfile):
    myAudio = wavfile
    try:
        samplingFreq, mySound = wavfile.read(myAudio)
    except:
        wav = wavio.read(myAudio);mySound=wav.data;samplingFreq=wav.rate
    mySoundDataType = mySound.dtype
    mySound = mySound / (2.**15)
    mySoundShape = mySound.shape
    mySoundOneChannel = mySound[:,0]
    return mySoundOneChannel, samplingFreq

def get_rms(tr):
    if tr==1:    return 43.26663389770346
    if tr==2:    return 44.59389653206161
    if tr==3:    return 46.40663592434644
    if tr==4:    return 54.60927837577519
    if tr==5:    return 39.21090720585492
    if tr==6:    return 49.662128408663705
    if tr==7:    return 53.75015494955412
    if tr==8:    return 44.507015861649144

for track in args.tracks:
    print("track : " + str(track))
    for sc in range(1,166):
        in_fdir = args.in_dir + "/scene%04d/"%sc
        out_fdir = args.out_dir + "/scene%04d/"%sc
        print(in_fdir)
        for g in glob.glob(in_fdir+"*.WAV"):
            audiofolder = g
        d1=audiofolder+"/"+audiofolder.split('/')[-1].split('.')[0]+"_Tr"+str(track)+".WAV";
        aud_Tr1,rate = get_signal(audiofolder+"/"+audiofolder.split('/')[-1].split('.')[0]+"_Tr"+str(track)+".WAV")
        rms_value = get_rms(track)
        aud_Tr1 = normalize(aud_Tr1,rms=rms_value)
        save_dir=out_fdir+"spectrograms_full/Track"+str(track)+"/";
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in tqdm(sorted(glob.glob(in_fdir+"split_videoframes_full/*.png"))):
            audio_start_time = int(i.split('_')[-1].split('.')[0])#random.uniform(0, 9.9 - self.opt.audio_length)
            audio_end_time = audio_start_time + audio_length
            audio_start = int(audio_start_time * rate)
            audio_end = audio_start + int(audio_length * rate)
            audio = aud_Tr1[audio_start:audio_end]
            #audio = normalize(audio)
            audio_channel1 = audio[:]
            audio_img = generate_spectrogram(audio_channel1)
            #print(audio_channel1.shape[0],audio_img.shape)
            if audio_channel1.shape[0]==96000:
                np.save(save_dir+'%06d.npy'%audio_start_time,audio_img)
    

