import torch
import torch.nn as nn

from utils.utils import Norm2d, Upsample, initialize_weights

class semanDecoder(nn.Module):
    def __init__(self,in_channel, out_channel, out_size=[1920, 3840]):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = Norm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = Norm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.out_size = out_size
        initialize_weights(self.conv1)
        initialize_weights(self.bn1)
        initialize_weights(self.conv2)
        initialize_weights(self.bn2)
        initialize_weights(self.conv3)

    def forward(self, x):
        x = Upsample(x, [int(self.out_size[0]/2), int(self.out_size[1]/2)])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        x = Upsample(x, self.out_size)
        
        return x