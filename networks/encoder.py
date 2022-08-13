import torch
import torch.nn as nn

from utils.utils import Norm2d, Upsample, initialize_weights

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

def downconv(in_channel, out_channel, norm_layer=nn.BatchNorm2d):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
    relu = nn.LeakyReLU(0.2, True)
    norm = norm_layer(out_channel)
    return nn.Sequential(*[conv, relu, norm])

class audioEncoder(nn.Module):
    def __init__(self, in_channel, mid_channel=64):
        super().__init__()

        self.conv1 = downconv(in_channel, mid_channel)
        self.conv2 = downconv(mid_channel, mid_channel * 2)
        self.conv3 = downconv(mid_channel * 2, mid_channel * 4)
        self.conv4 = downconv(mid_channel * 4, mid_channel * 8)
        
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 64, output_stride=8)
        self.post_us = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.post_cat = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.post_aspp = nn.Conv2d(320, 256, kernel_size=1, bias=False)

        initialize_weights(self.post_us)
        initialize_weights(self.post_cat)

        
    def forward(self, audio_1, audio_2):
        audio_1 = self.conv1(audio_1)
        audio_1 = self.conv2(audio_1)
        audio_1 = self.conv3(audio_1)
        audio_1 = self.conv4(audio_1)

        audio_2 = self.conv1(audio_2)
        audio_2 = self.conv2(audio_2)
        audio_2 = self.conv3(audio_2)
        audio_2 = self.conv4(audio_2)

        audio_1 = Upsample(audio_1, [60, 120])
        audio_1 = self.post_us(audio_1)
        audio_2 = Upsample(audio_2, [60, 120])
        audio_2 = self.post_us(audio_2)

        audio = torch.cat([audio_1, audio_2], 1)
        audio = self.post_cat(audio)

        audio = self.aspp(audio)
        audio = self.post_aspp(audio)

        return audio

class audioEncoderSeparate(nn.Module):
    def __init__(self, in_channel, mid_channel=64):
        super().__init__()

        self.conv1_audio1 = downconv(in_channel, mid_channel)
        self.conv2_audio1 = downconv(mid_channel, mid_channel * 2)
        self.conv3_audio1 = downconv(mid_channel * 2, mid_channel * 4)
        self.conv4_audio1 = downconv(mid_channel * 4, mid_channel * 8)

        self.conv1_audio2 = downconv(in_channel, mid_channel)
        self.conv2_audio2 = downconv(mid_channel, mid_channel * 2)
        self.conv3_audio2 = downconv(mid_channel * 2, mid_channel * 4)
        self.conv4_audio2 = downconv(mid_channel * 4, mid_channel * 8)
        
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 64, output_stride=8)
        self.post_us_audio1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.post_us_audio2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)

        self.post_cat = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.post_aspp = nn.Conv2d(320, 256, kernel_size=1, bias=False)

        initialize_weights(self.post_us_audio1)
        initialize_weights(self.post_us_audio2)
        initialize_weights(self.post_cat)

        
    def forward(self, audio_1, audio_2):
        audio_1 = self.conv1_audio1(audio_1)
        audio_1 = self.conv2_audio1(audio_1)
        audio_1 = self.conv3_audio1(audio_1)
        audio_1 = self.conv4_audio1(audio_1)

        audio_2 = self.conv1_audio2(audio_2)
        audio_2 = self.conv2_audio2(audio_2)
        audio_2 = self.conv3_audio2(audio_2)
        audio_2 = self.conv4_audio2(audio_2)

        audio_1 = Upsample(audio_1, [60, 120])
        audio_1 = self.post_us_audio1(audio_1)
        audio_2 = Upsample(audio_2, [60, 120])
        audio_2 = self.post_us_audio2(audio_2)

        audio = torch.cat([audio_1, audio_2], 1)
        audio = self.post_cat(audio)

        audio = self.aspp(audio)
        audio = self.post_aspp(audio)

        return audio


class singleAudioEncoder(nn.Module):
    def __init__(self, in_channel, mid_channel=64):
        super().__init__()

        self.conv1_audio1 = downconv(in_channel, mid_channel)
        self.conv2_audio1 = downconv(mid_channel, mid_channel * 2)
        self.conv3_audio1 = downconv(mid_channel * 2, mid_channel * 4)
        self.conv4_audio1 = downconv(mid_channel * 4, mid_channel * 8)

        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 64, output_stride=8)
        self.post_us_audio1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.post_aspp = nn.Conv2d(320, 256, kernel_size=1, bias=False)

        initialize_weights(self.post_us_audio1)

        
    def forward(self, audio_1):
        audio_1 = self.conv1_audio1(audio_1)
        audio_1 = self.conv2_audio1(audio_1)
        audio_1 = self.conv3_audio1(audio_1)
        audio_1 = self.conv4_audio1(audio_1)

        audio_1 = Upsample(audio_1, [60, 120])
        audio = self.post_us_audio1(audio_1)

        audio = self.post_cat(audio)

        audio = self.aspp(audio)
        audio = self.post_aspp(audio)

        return audio

"""
from torchsummary import summary
model = audioEncoder(2, 2)
model = model.to("cuda")
summary(model, [(2, 257, 601), (2, 257, 601)])
"""