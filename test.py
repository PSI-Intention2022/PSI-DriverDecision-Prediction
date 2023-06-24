import os
import torch
import json

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def validate_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)
        # intent_pred: logit output, bs
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)
            # gt_disagreement = data['disagree_score'][:, args.observe_length]
            # gt_consensus = (1 - gt_disagreement).to(device)

        recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                   intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

    recorder.eval_intent_epoch_calculate(writer)

    return recorder


def test_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    recorder.eval_epoch_reset(epoch, niters)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)
        # intent_pred: logit output, bs x 1
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)

        recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                   intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

    recorder.eval_intent_epoch_calculate(writer)

    return recorder


def predict_intent(model, dataloader, args):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)

        for i in range(len(data['frames'])):
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent
            # gt_int = data['intention_binary'][i][args.observe_length].item()  # int list, bs x 60
            # gt_int_prob = data['intention_prob'][i][args.observe_length].item()  # float list, bs x 60
            # gt_disgr = data['disagree_score'][i][args.observe_length].item()  # float list, bs x 60
            int_prob = intent_prob[i].item()
            int_pred = round(int_prob) # <0.5 --> 0, >=0.5 --> 1.

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['intent_pred'] = int_pred
            dt[vid][pid][fid]['intent_pred_prob'] = int_prob

    with open(os.path.join(args.checkpoint_path, 'results', 'test_intent_prediction.json'), 'w') as f:
        json.dump(dt, f)


def validate_traj(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        traj_pred = model(data)
        traj_gt = data['bboxes'][:, args.observe_length: , :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        # if args.normalize_bbox == 'subtract_first_frame':
        #     traj_pred = traj_pred + data['bboxes'][:, :1, :].type(FloatTensor)
        recorder.eval_traj_batch_update(itern, data, traj_gt.detach().cpu().numpy(), traj_pred.detach().cpu().numpy())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

    recorder.eval_traj_epoch_calculate(writer)

    return recorder


def predict_traj(model, dataloader, args):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        traj_pred = model(data)
        traj_gt = data['original_bboxes'][:, args.observe_length:, :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        # print("Prediction: ", traj_pred.shape)

        for i in range(len(data['frames'])): # for each sample in a batch
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['traj_pred'] = traj_pred[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    # print("saving prediction...")
    with open(os.path.join(args.checkpoint_path, 'results', 'test_traj_prediction.json'), 'w') as f:
        json.dump(dt, f)


@torch.no_grad()
def validate_driving(epoch, model, dataloader, args, recorder, writer):
    print(f"Validate ...")
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        pred_speed_logit, pred_dir_logit = model(data)
        lbl_speed = data['label_speed']  # bs x 1
        lbl_dir = data['label_direction']  # bs x 1
        recorder.eval_driving_batch_update(itern, data, lbl_speed.detach().cpu().numpy(), lbl_dir.detach().cpu().numpy(),
                                         pred_speed_logit.detach().cpu().numpy(), pred_dir_logit.detach().cpu().numpy())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

        del data
        del pred_speed_logit
        del pred_dir_logit

    recorder.eval_driving_epoch_calculate(writer)

    return recorder


@torch.no_grad()
def predict_driving(model, dataloader, args, dset='test'):
    print(f"Predict and save prediction of {dset} set...")
    model.eval()
    dt = {}
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        pred_speed_logit, pred_dir_logit = model(data)
        # lbl_speed = data['label_speed']  # bs x 1
        # lbl_dir = data['label_direction']  # bs x 1
        # print("batch size: ", len(data['frames']), len(data['video_id']))
        for i in range(len(data['frames'])): # for each sample in a batch
            # print(data['video_id'])
            vid = data['video_id'][0][i] # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if fid not in dt[vid]:
                dt[vid][fid] = {}
            dt[vid][fid]['speed'] = torch.argmax(pred_speed_logit[i]).item()
            dt[vid][fid]['direction'] = torch.argmax(pred_dir_logit[i]).item()

        if itern % args.print_freq == 10:
            print(f"Predicting driving decision of Batch {itern}/{niters}")
        del data
        del pred_speed_logit
        del pred_dir_logit

    print("Saving prediction to file...")
    with open(os.path.join(args.checkpoint_path, 'results', f'{dset}_driving_pred.json'), 'w') as f:
        json.dump(dt, f)

