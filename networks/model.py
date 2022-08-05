import torch
import torch.nn as nn
from .encoder import audioEncoder
from .decoder_seman import semanDecoder

class audioToSeman(nn.Module):
    def __init__(self, num_class, in_channel=1, out_size=[1920, 3840]):
        super().__init__()
        self.encoder = audioEncoder(in_channel=in_channel)
        self.decoder = semanDecoder(256, num_class, out_size=out_size)

    def forward(self, audio_1, audio_2):
        x = self.encoder(audio_1, audio_2)
        x = self.decoder(x)
        return x