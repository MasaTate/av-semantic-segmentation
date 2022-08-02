import torch.nn as nn

def Norm2d(in_channels):
    return nn.BatchNorm2d(in_channels)

def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()