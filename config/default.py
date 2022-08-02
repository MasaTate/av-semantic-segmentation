from yacs.config import CfgNode as CN

_C = CN()

# CUDA
_C.CUDA = CN()
_C.CUDA.USE_CUDA = True
_C.CUDA.CUDA_NUM = 0

# dataset
_C.DATASET = CN()

_C.DATASET.ROOT = "/work/masatate/dataset/dataset_public"
_C.DATASET.NUM_CLASSES = 3

# model
_C.MODEL = CN()
_C.MODEL.PRETRAINED = None

# log
_C.LOG = CN()
_C.LOG.DIR = './logs'
_C.LOG.LOSS = 20
_C.LOG.IMAGE = 2000

# result
_C.RESULT = CN()
_C.RESULT.PATH = "./result"
_C.RESULT.SAVE_NUM = 3
_C.RESULT.WEIGHT_PATH = "./checkpoint"

# training
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.EPOCH_START = 0
_C.TRAIN.EPOCH_END = 100
_C.TRAIN.VAL_EPOCH = 1



def get_cfg_defaults():
    return _C.clone()