import json
import os
from collections import namedtuple
from pickletools import uint8

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class SoundCityscapes(data.Dataset):
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
        CityscapesClass('road',                 7, 255, 'flat', 1, False, False, (128, 64, 128)), #train_id = 0
        CityscapesClass('sidewalk',             8, 255, 'flat', 1, False, False, (244, 35, 232)), #train_id = 1
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 255, 'construction', 2, False, False, (70, 70, 70)), #train_id = 2
        CityscapesClass('wall',                 12, 255, 'construction', 2, False, False, (102, 102, 156)), #train_id = 3
        CityscapesClass('fence',                13, 255, 'construction', 2, False, False, (190, 153, 153)), #train_id = 4
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 255, 'object', 3, False, False, (153, 153, 153)), #train_id = 5
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 255, 'object', 3, False, False, (250, 170, 30)), #train_id = 6
        CityscapesClass('traffic sign',         20, 255, 'object', 3, False, False, (220, 220, 0)), #train_id = 7
        CityscapesClass('vegetation',           21, 255, 'nature', 4, False, False, (107, 142, 35)), #train_id = 8
        CityscapesClass('terrain',              22, 255, 'nature', 4, False, False, (152, 251, 152)), #train_id = 9
        CityscapesClass('sky',                  23, 255, 'sky', 5, False, False, (70, 130, 180)), #train_id = 10
        CityscapesClass('person',               24, 255, 'human', 6, True, False, (220, 20, 60)), #train_id = 11
        CityscapesClass('rider',                25, 255, 'human', 6, True, False, (255, 0, 0)), #train_id = 12
        CityscapesClass('car',                  26, 0, 'vehicle', 7, True, False, (0, 0, 142)), #train_id = 13
        CityscapesClass('truck',                27, 255, 'vehicle', 7, True, False, (0, 0, 70)), #train_id = 14
        CityscapesClass('bus',                  28, 255, 'vehicle', 7, True, False, (0, 60, 100)), #train_id = 15
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 1, 'vehicle', 7, True, False, (0, 80, 100)), #train_id = 16
        CityscapesClass('motorcycle',           32, 2, 'vehicle', 7, True, False, (0, 0, 230)), #train_id = 17
        CityscapesClass('bicycle',              33, 255, 'vehicle', 7, True, False, (119, 11, 32)), #train_id = 18
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

    def __init__(self, root, split='train', mode='fine', target_type='color', transform=None, sound_track=[3, 8]):
        self.root = os.path.expanduser(root)
        self.mode = 'gtDLab'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'masked_videoframes', split)
        self.spectrogram_dir = self.root
        self.mask_dir = self.root

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []
        self.mask = []
        self.spectrogram_1 = []
        self.spectrogram_2 = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)


            for i, file_name in enumerate(os.listdir(img_dir)):
                if i > 1 and self.split == "val":
                    break
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(os.path.splitext(file_name)[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))
                self.mask.append(os.path.join(self.mask_dir, city, "pred_sound_making", os.path.basename(file_name)))
                number = os.path.splitext(os.path.basename(file_name))[0].split("_")[-1]
                self.spectrogram_1.append(os.path.join(self.spectrogram_dir, city, "spectrograms", "Track"+str(sound_track[0]), number+".npy"))
                self.spectrogram_2.append(os.path.join(self.spectrogram_dir, city, "spectrograms", "Track"+str(sound_track[1]), number+".npy"))
                exist = os.path.exists(self.targets[-1]) and os.path.exists(self.mask[-1]) and os.path.exists(self.spectrogram_1[-1]) and os.path.exists(self.spectrogram_2[-1])
                if not exist:
                    self.images.pop(-1)
                    self.targets.pop(-1)
                    self.mask.pop(-1)
                    self.spectrogram_1.pop(-1)
                    self.spectrogram_2.pop(-1)
                assert len(self.images) > 0
                assert len(self.images) == len(self.mask) == len(self.targets) == len(self.spectrogram_1) == len(self.spectrogram_2)


    @classmethod
    def encode_color(cls, target, mask):
        target = np.array(target)
        mask = np.array(mask)
        assert target.shape[:2] == mask.shape
        train_id = np.full((target.shape[0], target.shape[1]), 255, dtype=np.uint8)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if mask[i][j] != 0:
                    if len(np.where((cls.train_id_to_color == target[i][j]).all(axis=1))[0]) != 0:
                        train_id[i][j] = np.where((cls.train_id_to_color == target[i][j]).all(axis=1))[0][0]
                        #print(train_id[i][j])
        #train_id[train_id == len(cls.train_id_to_color)-1] = 255
        #print("color encode")
        #print(train_id[(train_id > 2) & (train_id != 255)])
        train_id[train_id == 255] = len(cls.train_id_to_color) - 1
        return torch.from_numpy(train_id)

    @classmethod
    def encode_color_fast(cls, target, mask):
        target = np.array(target)
        mask = np.array(mask)
        assert target.shape[:2] == mask.shape
        train_id = np.full((target.shape[0], target.shape[1]), 255, dtype=np.uint8)
        color_dict = cls.train_id_to_color.sum(axis=1)
        target = target.sum(axis=2)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if mask[i][j] != 0:
                    if len(np.where((color_dict == target[i][j]))[0]) != 0:
                        train_id[i][j] = np.where((color_dict == target[i][j]))[0][0]
                        #print(train_id[i][j])
        #train_id[train_id == len(cls.train_id_to_color)-1] = 255
        train_id[train_id == 255] = len(cls.train_id_to_color) - 1
        return torch.from_numpy(train_id)
        

    
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        #target[target == 255] = len(cls.train_id_to_color) - 1
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
        #print("load")
        image = Image.open(self.images[index]).convert('RGB')
        #image_black = Image.new("RGB", (3840, 1920))
        target = Image.open(self.targets[index]).convert('RGB')
        mask = Image.open(self.mask[index]).convert('L')
        spec_1 = np.load(self.spectrogram_1[index])
        spec_2 = np.load(self.spectrogram_2[index])
        #print("transform")
        if self.transform:
            image, target, mask = self.transform(image, target, mask)
            #image_black, target, mask = self.transform(image_black, target, mask)
        #target = self.encode_target(target)
        #print("encode")
        target = self.encode_color_fast(target, mask)
        #print("check")
        #print(target[(target > 2) & (target != 255)])
        return image, target, torch.from_numpy(spec_1), torch.from_numpy(spec_2)

    def __len__(self):
        return len(self.images)

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