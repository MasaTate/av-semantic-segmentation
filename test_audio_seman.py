from webbrowser import get
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.default import get_cfg_defaults
import dataset
import networks
from metrics.metrics import SegMetrics
from optimizer import get_optimizer
import utils.visual_transform as vt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import loss as Loss

def load_config(config_path=None):
    cfg = get_cfg_defaults()
    if config_path is not None:
        cfg.merge_from_file(config_path)
    cfg.freeze()

    return cfg

def main():
    cfg = load_config("./config/test.yaml")
    # device
    device = torch.device('cuda:'+str(cfg.CUDA.CUDA_NUM) if torch.cuda.is_available() and cfg.CUDA.USE_CUDA==True else 'cpu')
    print("device:"+str(device))

    # prepare dataset
    print("loading dataset...")
    test_transforms = vt.PairCompose([vt.PairToTensor()])
    test_data = dataset.get_dataset(type=cfg.DATASET.TYPE, root=cfg.DATASET.ROOT, split='val', transform=test_transforms, sound_track=cfg.DATASET.TRACK)

    test_loader = DataLoader(test_data, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS)

    # prepare model
    print("loading_model...")
    model = networks.get_model(type=cfg.MODEL.TYPE, num_class=cfg.DATASET.NUM_CLASSES, in_channel=1)
    if cfg.MODEL.PRETRAINED is None: print("Input the model weight")
    
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    model = model.to(device)

    # result
    if not os.path.exists(cfg.RESULT.PATH):
        os.makedirs(cfg.RESULT.PATH)
        print("created : "+cfg.RESULT.PATH)
    metrics = SegMetrics(cfg.DATASET.NUM_CLASSES, device)

    score = validation(test_loader, model, device, metrics, cfg.RESULT.PATH, cfg.RESULT.SAVE_NUM)
    print(score)


def validation(val_loader, model, device, metrics, results_path, save_num):
    model.eval()
    print("======================evaluation======================")
    save_count = 0
    for i, (image, target, audio_1, audio_2) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            image = image.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)
            audio_1 = audio_1.to(device, dtype=torch.float32)
            audio_2 = audio_2.to(device, dtype=torch.float32)

            pred = model(audio_1, audio_2, target)
            pred_label = pred.detach().max(dim=1)[1]

            metrics.update(target, pred_label)

            
            if save_num != 0 and i % (len(val_loader) // save_num) == 0:
                    image_save = image[0].detach().cpu().numpy()
                    target_save = target[0].cpu().numpy()
                    pred_save = pred_label[0].cpu().numpy()
                    
                    image_save = (image_save*255).transpose(1, 2, 0).astype(np.uint8)
                    target_save = val_loader.dataset.decode_target(target_save).astype(np.uint8)
                    pred_save = val_loader.dataset.decode_target(pred_save).astype(np.uint8)
                    
                    Image.fromarray(image_save).save(results_path+"/image_{}.png".format(save_count))
                    Image.fromarray(target_save).save(results_path+"/label_{}.png".format(save_count))
                    Image.fromarray(pred_save).save(results_path+"/predict_{}.png".format(save_count))
                    
                    save_count += 1
            
            del image, target, audio_1, audio_2
            
    score = metrics.get_results()

    return score

if __name__ == "__main__":
    main()