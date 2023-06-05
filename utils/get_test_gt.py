from opts import get_opts
from datetime import datetime
import os
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
# from train import train_intent
from test import validate_intent, test_intent
from utils.log import RecordResults


def main(args):
    writer = SummaryWriter(args.checkpoint_path)
    recorder = RecordResults(args)
    ''' 1. Load database '''
    if not os.path.exists(os.path.join(args.database_path, 'intent_database_train.pkl')):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)
    get_test_driving_gt(test_loader, '../test_gt/test_driving_gt.json', args)

def get_test_driving_gt(model, dataloader, args, dset='test'):
    dt = {}
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        lbl_speed = data['label_speed']  # bs x 1
        lbl_dir = data['label_direction']  # bs x 1
        for i in range(len(data['frames'])):  # for each sample in a batch
            # print(data['video_id'])
            vid = data['video_id'][0][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if fid not in dt[vid]:
                dt[vid][fid] = {}
            dt[vid][fid]['speed'] = lbl_speed[i].item()
            dt[vid][fid]['direction'] = lbl_dir[i].item()

        if itern % args.print_freq == 0:
            print(f"Get gt driving decision of Batch {itern}/{niters}")

        # if itern >= 10:
        #     break
    with open(os.path.join(f'./test_gt/{dset}_driving_gt.json'), 'w') as f:
        json.dump(dt, f)

if __name__ == '__main__':
    args = get_opts()
    # Task
    args.datset = 'PSI200'
    args.task_name = 'ped_intent'
    args.model_name = 'lstm_int_bbox' # LSTM module, with bboxes sequence as input, to predict intent

    # Model
    args.load_image = False # only bbox input
    if args.load_image:
        args.backbone = 'resnet'
        args.freeze_backbone = False
    else:
        args.backbone = None
        args.freeze_backbone = False
    args.loss_weights = {
        'loss_intent': 1.0,
        'loss_traj': 1.0,
        'loss_driving': 1.0
    }
    # Data - intent prediction
    args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
    args.intent_type = 'mean'
    args.intent_loss = ['bce']
    args.intent_disagreement = 1 # -1: not use disagreement 1: use disagreement to reweigh samples
    args.intent_positive_weight = 0.5 # reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)

    # Data - trajectory
    args.traj_loss = ['bbox_l1']
    args.normalize_bbox = None
    # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
    # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]

    # Train
    args.epochs = 10
    args.batch_size = 128
    args.lr = 1e-3
    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    # Record
    now = datetime.now()
    time_folder = now.strftime('%Y%m%d%H%M%S')
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name, args.dataset, args.model_name, time_folder)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    main(args)