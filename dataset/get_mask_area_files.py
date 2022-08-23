import numpy as np
import cv2
from tqdm import tqdm
import glob


mdict = {}
thresh = 5
for sc in tqdm(range(1,166)):
    fdir="/work/masatate/dataset/dataset_public/scene%04d/"%sc
    mdict[sc]=[]
    #print(sc,len(glob.glob(fdir+f"wavsplits/Track{track}/"+"*.npy")))
    count = 0
    for g in glob.glob(fdir+"binary_mask_full/*_mask.png"):
        img = cv2.imread(g)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        assert len(img.shape) == 2
        pixel = img.shape[0] * img.shape[1]
        mask_rate = img.sum()/255/pixel*100
        print(mask_rate)
        if mask_rate>=thresh:
            mdict[sc].append(fdir+"binary_mask_full/"+g.split('/')[-1].split('.')[0]+".png")
            count+=1
    print("scene:"+str(sc),"count:"+str(count))
    #scene_energy = np.array(scene_energy)
    #if scene_energy.max() < 1e-4:
        #print("Track : %d, scene : %d"%(track, sc))
np.save(f'/work/masatate/dataset/dataset_public/MaskArea_165scenes_{thresh}percent.npy',mdict)
print(f"Saved Area File.")

### To plot the energy plot ###
"""
energyArr = np.load('/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/audio/dataset/SoundEnergy_165scenes.npy')
print(len(energyArr))
energyArr=[i for i in energyArr if i<0.1]
n, bins, patches = plt.hist(energyArr,'auto',density=True, facecolor='g', alpha=0.75)

num1 = sum(i > thresh for i in energyArr)
num0 = sum(i < thresh for i in energyArr)
print("Percent > thresh: %f"% (num1/float(num1+num0)))
print("Percent < thresh: %f"% (num0/float(num1+num0)))
plt.title("Audio Energy distribution of our dataset samples",fontsize=20)
plt.xlabel("Sound Energy (~Hz^-2)",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.xlim(0.0, 0.02)
plt.tick_params(axis="x", labelsize=18)
plt.tick_params(axis="y", labelsize=18)
plt.grid(True)
plt.show()
"""
