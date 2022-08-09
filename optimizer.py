import math
import torch
import torch.optim as optim

def get_optimizer(model, lr=0.00001, optimizer='Adam', scheduler='poly'):
    if optimizer == 'Adam':
        opt = optim.Adam([{'params':model.encoder.parameters(), 'lr':lr*5}, {'params':model.decoder.parameters(), 'lr':lr}], lr=lr, weight_decay=1e-4)
    else:
        raise ValueError("Not a valid optimizer")

    if scheduler == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / 180, 1)
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schdule')

    return opt, sch