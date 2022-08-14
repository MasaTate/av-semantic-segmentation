import torch
import torch.nn as nn
from .encoder import audioEncoder, audioEncoderSeparate, singleAudioEncoder
from .decoder_seman import semanDecoder

class audioToSeman(nn.Module):
    def __init__(self, num_class, in_channel=1):
        super().__init__()
        self.encoder = audioEncoder(in_channel=in_channel)
        self.decoder = semanDecoder(256, num_class)

    def forward(self, audio_1, audio_2, target):
        x = self.encoder(audio_1, audio_2)
        x = self.decoder(x, target)
        return x

class audioToSemanSep(nn.Module):
    def __init__(self, num_class, in_channel=1):
        super().__init__()
        self.encoder = audioEncoderSeparate(in_channel=in_channel)
        self.decoder = semanDecoder(256, num_class)

    def forward(self, audio_1, audio_2, target):
        x = self.encoder(audio_1, audio_2)
        x = self.decoder(x, target)
        return x

class singleAudioToSeman(nn.Module):
    def __init__(self, num_class, in_channel=1):
        super().__init__()
        self.encoder = singleAudioEncoder(in_channel=in_channel)
        self.decoder =semanDecoder(256, num_class)

    def forward(self, audio, target):
        x = self.encoder(audio)
        x = self.decoder(x, target)
        return x