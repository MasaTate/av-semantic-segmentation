from typing import Tuple
import torch
import torch.nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np


class PairResize:
    def __init__(self, size):
        assert len(size) == 2
        self.size = size
        self.resize = T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)

    def __call__(self, img, trg, msk):
        return self.resize(img), self.resize(trg), self.resize(msk)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, trg, msk):
        for t in self.transforms:
            img, trg, msk = t(img, trg, msk)
        return img, trg, msk
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '     {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class PairToTensor:
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize=normalize
        self.target_type = target_type
        
    def __call__(self, img, trg, msk):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor. 
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            return F.to_tensor(img), torch.from_numpy(np.array(trg, dtype=self.target_type)), torch.from_numpy(np.array(msk, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)), torch.from_numpy(np.array(trg, dtype=self.target_type)), torch.from_numpy(np.array(msk, dtype=self.target_type))
        
    def __repr__(self):
        return self.__class__.__name__ + '()'