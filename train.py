import collections

from test import validate_traj, validate_driving
import torch
import numpy as np
import os

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def train_driving(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer):
    pos_weight = torch.tensor(args.intent_positive_weight).to(device) # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight).to(device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(device),
        'BCELoss': torch.nn.BCELoss().to(device),
        'CELoss': torch.nn.CrossEntropyLoss().to(device),
        'L1Loss': torch.nn.L1Loss().to(device),
    }
    epoch_loss = {'loss_driving': [], 'loss_driving_speed': [], 'loss_driving_dir': []}

    for epoch in range(1, args.epochs + 1):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_driving_epoch(epoch, model, optimizer, criterions, epoch_loss, train_loader, args, recorder, writer)
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                  f"loss_driving_speed = {np.mean(epoch_loss['loss_driving_speed']): .4f}, "
                  f"loss_driving_dir = {np.mean(epoch_loss['loss_driving_dir']): .4f}")

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            validate_driving(epoch, model, val_loader, args, recorder, writer)

        torch.save(model.state_dict(), args.checkpoint_path + f'/latest.pth')


def train_driving_epoch(epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        optimizer.zero_grad()
        pred_speed_logit, pred_dir_logit = model(data)
        lbl_speed = data['label_speed'].type(LongTensor)  # bs x 1
        lbl_dir = data['label_direction'].type(LongTensor)  # bs x 1
        # traj_pred = model(data)
        # intent_pred: sigmoid output, (0, 1), bs
        # traj_pred: logit, bs x ts x 4

        # traj_gt = data['bboxes'][:, args.observe_length:, :].type(FloatTensor)
        # bs, ts, _ = traj_gt.shape
        # center: bs x ts x 2
        # traj_center_gt = torch.cat((((traj_gt[:, :, 0] + traj_gt[:, :, 2]) / 2).unsqueeze(-1),
        #                             ((traj_gt[:, :, 1] + traj_gt[:, :, 3]) / 2).unsqueeze(-1)), dim=-1)
        # traj_center_pred = torch.cat((((traj_pred[:, :, 0] + traj_pred[:, :, 2]) / 2).unsqueeze(-1),
        #                               ((traj_pred[:, :, 1] + traj_pred[:, :, 3]) / 2).unsqueeze(-1)), dim=-1)

        loss_driving = torch.tensor(0.).type(FloatTensor)
        if 'cross_entropy' in args.driving_loss:
            loss_driving_speed = torch.mean(criterions['CELoss'](pred_speed_logit, lbl_speed))
            loss_driving_dir = torch.mean(criterions['CELoss'](pred_dir_logit, lbl_dir))
            # loss_bbox_l1 = torch.mean(criterions['L1Loss'](traj_pred, traj_gt))
            batch_losses['loss_driving_speed'].append(loss_driving_speed.item())
            batch_losses['loss_driving_dir'].append(loss_driving_dir.item())
            loss_driving += loss_driving_speed
            loss_driving += loss_driving_dir

        loss = args.loss_weights['loss_driving'] * loss_driving
        loss.backward()
        optimizer.step()

        # Record results
        batch_losses['loss'].append(loss.item())
        batch_losses['loss_driving'].append(loss_driving.item())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                  f"loss_driving_speed = {np.mean(batch_losses['loss_driving_speed']): .4f}, "
                  f"loss_driving_dir = {np.mean(batch_losses['loss_driving_dir']): .4f}")
        recorder.train_driving_batch_update(itern, data, lbl_speed.detach().cpu().numpy(), lbl_dir.detach().cpu().numpy(),
                                         pred_speed_logit.detach().cpu().numpy(), pred_dir_logit.detach().cpu().numpy(), loss.item(),
                                            loss_driving_speed.item(), loss_driving_dir.item())

        # if itern >= 10:
        #     break

    epoch_loss['loss_driving'].append(np.mean(batch_losses['loss_driving']))
    epoch_loss['loss_driving_speed'].append(np.mean(batch_losses['loss_driving_speed']))
    epoch_loss['loss_driving_dir'].append(np.mean(batch_losses['loss_driving_dir']))

    recorder.train_driving_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f'LearningRate', optimizer.param_groups[-1]['lr'], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f'Losses/{key}', np.mean(val), epoch)

    return epoch_loss