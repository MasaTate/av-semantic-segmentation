import json
import os
from collections import namedtuple
from pickletools import uint8
import glob

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

def make_dataset(root, mode, tracks=[3, 8], check_track=3):
    """
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    check_track = 3
    audioDict=np.load(os.path.join(root, f"SoundEnergy_165scenes_Track{check_track}.npy"), allow_pickle=True) #np.load('/'.join(root.split('/')[:-1])+"/SoundEnergy_165scenes_Track1.npy", allow_pickle=True)
    assert (mode in ['train', 'val'])
    if mode == 'train':
        for sc in tqdm(range(1,115)):
            img_dir_name = 'scene%04d'%sc
            check_audioImg_path = os.path.join(root, img_dir_name, f'spectrograms_full/Track{check_track}/')
            # image
            img_path = os.path.join(root, img_dir_name,'split_videoframes_full/')
            
            # spectrogram
            """
            randomno = np.random.randint(0,2)
            if randomno ==0:
                audioImg_path = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[0])}/')
                audioImg_path6 = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[1])}/')
            else:
                audioImg_path = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[1])}/')
                audioImg_path6 = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[0])}/')
            """
            audioImg_path = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[0])}/')
            audioImg_path6 = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[1])}/')
            
            # semantic segmentation ground truth
            mask_path = os.path.join(root, img_dir_name,'gtDLab/')
            
            # sound making map
            binary_mask_path = os.path.join(root, img_dir_name,'binary_mask_full/')
            
            seg_postfix = '_gtDLab_labelIds.png'
            mask_postfix = '_mask.png'
            for it_full in glob.glob(check_audioImg_path+"*.npy"):
                #print(it_full)
                it = it_full.split('/')[-1].split('.')[0]
                if it_full in audioDict.item()[sc]:
                    assert len(glob.glob(os.path.join(img_path, "*_"+it+".png"))) == 1
                    assert len(glob.glob(os.path.join(mask_path, "*"+it+seg_postfix))) == 1
                    assert len(glob.glob(os.path.join(binary_mask_path, "*"+it+mask_postfix))) == 1
                    item = (glob.glob(os.path.join(img_path, "*_"+it+".png"))[0], glob.glob(os.path.join(mask_path, "*"+it+seg_postfix))[0], glob.glob(os.path.join(binary_mask_path, "*"+it+mask_postfix))[0], os.path.join(audioImg_path, it+".npy"), os.path.join(audioImg_path6, it+".npy"), mode)
                    items.append(item)

    if mode == 'val':
        for sc in tqdm(range(115,140)):
            img_dir_name = 'scene%04d'%sc
            check_audioImg_path = os.path.join(root, img_dir_name, f'spectrograms_full/Track{check_track}/')
            # image
            img_path = os.path.join(root, img_dir_name,'split_videoframes_full/')
            
            # spectrogram
            audioImg_path = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[0])}/')
            audioImg_path6 = os.path.join(root, img_dir_name, f'spectrograms_full/Track{int(tracks[1])}/')
            
            # semantic segmentation ground truth
            mask_path = os.path.join(root, img_dir_name,'gtDLab/')
            
            # sound making map
            binary_mask_path = os.path.join(root, img_dir_name,'binary_mask_full/')
            
            seg_postfix = '_gtDLab_labelIds.png'
            mask_postfix = '_mask.png'
            for it_full in glob.glob(check_audioImg_path+"*.npy"):
                #print(it_full)
                it = it_full.split('/')[-1].split('.')[0]
                if it_full in audioDict.item()[sc]:
                    assert len(glob.glob(os.path.join(img_path, "*_"+it+".png"))) == 1
                    assert len(glob.glob(os.path.join(mask_path, "*"+it+seg_postfix))) == 1
                    assert len(glob.glob(os.path.join(binary_mask_path, "*"+it+mask_postfix))) == 1
                    item = (glob.glob(os.path.join(img_path, "*_"+it+".png"))[0], glob.glob(os.path.join(mask_path, "*"+it+seg_postfix))[0], glob.glob(os.path.join(binary_mask_path, "*"+it+mask_postfix))[0], os.path.join(audioImg_path, it+".npy"), os.path.join(audioImg_path6, it+".npy"), mode)
                    items.append(item)

    return items

class SoundCityscapesDifferentFixedRotate(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)), #train_id = 0
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)), #train_id = 1
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)), #train_id = 2
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)), #train_id = 3
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)), #train_id = 4
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)), #train_id = 5
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)), #train_id = 6
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)), #train_id = 7
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)), #train_id = 8
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)), #train_id = 9
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)), #train_id = 10
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)), #train_id = 11
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)), #train_id = 12
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)), #train_id = 13
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)), #train_id = 14
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)), #train_id = 15
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)), #train_id = 16
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)), #train_id = 17
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)), #train_id = 18
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)), 
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='color', transform=None, sound_track=[3, 8], check_track=3, rotate=None):
        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')
    
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.items = make_dataset(root, split, sound_track, check_track)
        if len(self.items) == 0:
            raise RuntimeError("Found 0 images, please check the data set (or path)")

        self.rotate = rotate
        if self.rotate is not None:
            assert isinstance(self.rotate, int)
            assert self.rotate <= 360 and self.rotate >= 0
        

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = len(cls.train_id_to_color) - 1
        target[target == 0] = len(cls.train_id_to_color) - 1
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        img_path, mask_path, bin_mask_path, spec_path_1, spec_path_2, mode = self.items[index]

        image = Image.open(img_path).convert('RGB')
        target = Image.open(mask_path).convert('L')
        mask = Image.open(bin_mask_path).convert('L')
        spec_1 = np.load(spec_path_1)
        spec_2 = np.load(spec_path_2)

        if self.transform:
            image, target, mask = self.transform(image, target, mask)

        target = self.encode_target(target)
        target[mask<128] = 0
        target = torch.from_numpy(target)
        """
        if mode=='val':
            assert image.shape[-2] == 480
            image_rot = image
            target_rot = target
            mask_rot = mask
            image_rot[:,0:240,:] = image[:,240:480,:]; image_rot[:,240:480,:] = image[:,0:240,:]
            target_rot[:,0:240,:] = target[:,240:480,:]; target_rot[:,240:480,:] = target[:,0:240,:]
            mask_rot[:,0:240,:] = mask[:,240:480,:]; mask_rot[:,240:480,:] = mask[:,0:240,:]
            image = image_rot
            target = target_rot
            mask = mask_rot
        """
        # left rotation
        if self.rotate is not None:
            height, width = image.shape[-2:]
            print("height:",height, "width:", width)
            x = int(width * self.rotate / 360)
            image_rot = image.clone()
            target_rot = target.clone()
            mask_rot = mask.clone()
            image_rot[:,:,0:width-x] = image[:,:,x:width]; image_rot[:,:,width-x:width] = image[:,:,0:x]
            target_rot[:,0:width-x] = target[:,x:width]; target_rot[:,width-x:width] = target[:,0:x]
            mask_rot[:,0:width-x] = mask[:,x:width]; mask_rot[:,width-x:width] = mask[:,0:x]
            image = image_rot
            target = target_rot
            mask = mask_rot

            
        spec_1 = spec_1[0,:,:]**2 + spec_1[1,:,:]**2
        spec_2 = spec_2[0,:,:]**2 + spec_2[1,:,:]**2
        spec_1[spec_1<1e-5]=1e-5
        spec_2[spec_2<1e-5]=1e-5
        spec_1 = np.log(spec_1)
        spec_2 = np.log(spec_2)
        spec_1 = np.expand_dims(spec_1, axis=0)
        spec_2 = np.expand_dims(spec_2, axis=0)
        return image, target, torch.from_numpy(spec_1), torch.from_numpy(spec_2)

    def __len__(self):
        return len(self.items)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)