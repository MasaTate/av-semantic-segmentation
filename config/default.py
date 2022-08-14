from yacs.config import CfgNode as CN

_C = CN()

# CUDA
_C.CUDA = CN()
_C.CUDA.USE_CUDA = True
_C.CUDA.CUDA_NUM = 0

# dataset
_C.DATASET = CN()
_C.DATASET.TYPE = "SoundCityscapes"
_C.DATASET.ROOT = "/work/masatate/dataset/dataset_public"
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRACK = [1, 6]
_C.DATASET.CHECK_TRACK = 1
_C.DATASET.ROTATE = 0
_C.DATASET.CROP = None

# model
_C.MODEL = CN()
_C.MODEL.TYPE = "audioToSeman" #"audioToSemanSep"
_C.MODEL.PRETRAINED = None 


# log
_C.LOG = CN()
_C.LOG.DIR = './logs_newloss_lr'
_C.LOG.LOSS = 20
_C.LOG.IMAGE = 200

# result
_C.RESULT = CN()
_C.RESULT.PATH = "./result_newloss_lr"
_C.RESULT.SAVE_NUM = 100
_C.RESULT.WEIGHT_PATH = "./checkpoint_newloss_lr"
_C.RESULT.WEIGHT_ITER = 800

# training
_C.TRAIN = CN()
_C.TRAIN.LR = 0.00004
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.EPOCH_START = 0
_C.TRAIN.EPOCH_END = 20
_C.TRAIN.VAL_EPOCH = 2



def get_cfg_defaults():
    return _C.clone()