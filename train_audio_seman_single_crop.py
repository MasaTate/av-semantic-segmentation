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
import argparse

def load_config(config_path=None):
    cfg = get_cfg_defaults()
    if config_path is not None:
        cfg.merge_from_file(config_path)
    cfg.freeze()

    return cfg

def main(args):
    cfg = load_config(args.config)
    # device
    device = torch.device('cuda:'+str(cfg.CUDA.CUDA_NUM) if torch.cuda.is_available() and cfg.CUDA.USE_CUDA==True else 'cpu')
    print("device:"+str(device))

    # prepare dataset
    print("loading dataset...")
    train_transforms = vt.PairCompose([vt.PairToTensor()])
    val_transforms = vt.PairCompose([vt.PairToTensor()])
    train_data = dataset.get_dataset(type=cfg.DATASET.TYPE, root=cfg.DATASET.ROOT, split='train', transform=train_transforms, sound_track=cfg.DATASET.TRACK, check_track=cfg.DATASET.CHECK_TRACK, rotate=cfg.DATASET.ROTATE, crop=cfg.DATASET.CROP)
    val_data = dataset.get_dataset(type=cfg.DATASET.TYPE, root=cfg.DATASET.ROOT, split='val', transform=val_transforms, sound_track=cfg.DATASET.TRACK, check_track=cfg.DATASET.CHECK_TRACK, rotate=cfg.DATASET.ROTATE, crop=cfg.DATASET.CROP)

    train_loader = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
    print("---> " + cfg.DATASET.TYPE + " is loaded")

    # prepare model
    print("loading_model...")
    model = networks.get_model(type=cfg.MODEL.TYPE, num_class=cfg.DATASET.NUM_CLASSES, in_channel=1)

    if cfg.MODEL.PRETRAINED is not None:
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    model = model.to(device)
    if cfg.MODEL.PRETRAINED is not None:
            print("---> " + cfg.MODEL.TYPE + " is loaded (Pretrained)")
    else:
        print("---> " + cfg.MODEL.TYPE + " is loaded")

    # optimizer, scheduler
    optimizer, scheduler = get_optimizer(model, cfg.TRAIN.LR, optimizer="Adam", scheduler="poly")

    # logging
    if not os.path.exists(cfg.LOG.DIR):
        os.makedirs(cfg.LOG.DIR)
        print("created : "+cfg.LOG.DIR)
    writer = SummaryWriter(log_dir=cfg.LOG.DIR)

    # result
    if not os.path.exists(cfg.RESULT.PATH):
        os.makedirs(cfg.RESULT.PATH)
        print("created : "+cfg.RESULT.PATH)
    if not os.path.exists(cfg.RESULT.WEIGHT_PATH):
        os.makedirs(cfg.RESULT.WEIGHT_PATH)
        print("created : "+cfg.RESULT.WEIGHT_PATH)

    # criterion
    #criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    #criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion, _ = Loss.get_loss("weighted", cfg.DATASET.NUM_CLASSES, ignore_index=255, device=device)

    # metrics
    metrics = SegMetrics(cfg.DATASET.NUM_CLASSES, device)

    print("======================start training======================")

    for epoch in range(cfg.TRAIN.EPOCH_START, cfg.TRAIN.EPOCH_END):
        print(f'epoch : {epoch}')
        for i, (_, target, audio_1) in enumerate(tqdm(train_loader)):
            model.train()
            step = epoch * len(train_loader) + i

            target = target.to(device, dtype=torch.long)
            audio_1 = audio_1.to(device, dtype=torch.float32)
            #print("calc")
            optimizer.zero_grad()

            pred = model(audio_1, target)
            loss = criterion(pred, target)

            # back prop
            loss = loss.mean()
            loss.backward()

            clip = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            #print("fin")
            if (i+1) % cfg.LOG.LOSS == 0:
                np_loss = loss.detach().cpu().numpy()
                writer.add_scalar('train_loss', np_loss, step + 1)

            if (i+1) % cfg.LOG.IMAGE == 0:
                target_save = target[0].detach().cpu().numpy()
                pred_save = pred.detach().max(dim=1)[1].cpu().numpy()[0]
                target_save = train_loader.dataset.decode_target(target_save).astype(np.uint8)
                target_save = torch.from_numpy(target_save.astype(np.float32)).clone().permute(2, 0, 1)
                pred_save = train_loader.dataset.decode_target(pred_save).astype(np.uint8)
                pred_save = torch.from_numpy(pred_save.astype(np.float32)).clone().permute(2, 0, 1)
                writer.add_image('train_target', target_save, step + 1)
                writer.add_image('train_pred', pred_save, step + 1)

            del target, audio_1

            if (step + 1) % cfg.RESULT.WEIGHT_ITER == 0:
                print("saving weight...")
                torch.save(model.state_dict(), cfg.RESULT.WEIGHT_PATH + f'/checkpoint_epoch{epoch}_iter{step + 1}.pth')

        scheduler.step()    
            #TODO ずらす
            
        if (epoch+1) % cfg.TRAIN.VAL_EPOCH == 0:
            score = validation(val_loader, model, device, metrics, epoch, cfg.RESULT.PATH, cfg.RESULT.SAVE_NUM)
            print(score)
            writer.add_scalar('val_mIoU', score["Mean IoU"], epoch)
            writer.add_scalars('val_class_IoU', score["Class IoU Str"], epoch)
            
        torch.save(model.state_dict(), cfg.RESULT.WEIGHT_PATH + f'/checkpoint_epoch{epoch}.pth')
    writer.close()


def validation(val_loader, model, device, metrics, epoch, results_path, save_num):
    model.eval()
    print("======================evaluation======================")
    save_count = 0
    for i, (image, target, audio_1) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            image = image.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)
            audio_1 = audio_1.to(device, dtype=torch.float32)

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
                    
                    Image.fromarray(image_save).save(results_path+"/image_epoch{}_{}.png".format(epoch, save_count))
                    Image.fromarray(target_save).save(results_path+"/label_epoch{}_{}.png".format(epoch, save_count))
                    Image.fromarray(pred_save).save(results_path+"/predict_epoch{}_{}.png".format(epoch, save_count))
                    
                    save_count += 1
            
            del image, target, audio_1
            
    score = metrics.get_results()

    model.train()
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="additional configuration", default=None)
    args = parser.parse_args()
    main(args)