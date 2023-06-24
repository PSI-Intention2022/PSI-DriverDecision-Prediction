import os
import numpy as np
from utils.utils import AverageMeter
from utils.metrics import evaluate_driving
import json

class RecordResults():
    def __init__(self, args=None, intent=False, traj=False, driving=True, reason=False, extract_prediction=False):
        self.args = args
        self.save_output = extract_prediction
        self.reason = reason
        self.driving = driving
        self.intent = intent
        self.traj = traj

        self.all_train_results = {}
        self.all_eval_results = {}
        self.all_val_results = {}

        # cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        self.result_path = os.path.join(self.args.checkpoint_path, 'results')
        if not os.path.isdir(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)

        self._log_file = os.path.join(self.args.checkpoint_path, 'log.txt')
        open(self._log_file, 'w').close()

    def train_epoch_reset(self, epoch, nitern):
        # 1. initialize log info
        # (1.1) loss log list
        self.log_loss_total = AverageMeter()
        self.log_loss_driving_speed = AverageMeter()
        self.log_loss_driving_dir = AverageMeter()
        # (1.2) driving
        self.driving_speed_gt = []
        self.driving_speed_pred = []
        self.driving_dir_gt = []
        self.driving_dir_pred = []
        # (1.3) store all results
        self.train_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

    def train_driving_batch_update(self, itern, data, speed_gt, direction_gt, speed_pred_logit, dir_pred_logit,
                                   loss, loss_driving_speed, loss_driving_dir):
        # 3. Update training info
        # (3.1) loss log list
        bs = speed_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_driving_speed.update(loss_driving_speed, bs)
        self.log_loss_driving_dir.update(loss_driving_dir, bs)
        # (3.2) training data info

        self.driving_speed_gt.extend(speed_gt)  # bs
        self.driving_dir_gt.extend(direction_gt)
        self.driving_speed_pred.extend(np.argmax(speed_pred_logit, axis=-1))  # bs
        self.driving_dir_pred.extend(np.argmax(dir_pred_logit, axis=-1))  # bs


        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", 'a') as f:
                f.write('Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  driving speed Loss: {:.4f} |  driving dir Loss: {:.4f} \n'.format(
                    self.epoch, self.args.epochs, itern, self.nitern, self.log_loss_total.avg,
                    self.log_loss_driving_speed.avg, self.log_loss_driving_dir.avg))


    def train_driving_epoch_calculate(self, writer=None):
        print('----------- Training results: ------------------------------------ ')
        if self.driving:
            driving_results = evaluate_driving(np.array(self.driving_speed_gt), np.array(self.driving_dir_gt),
                                             np.array(self.driving_speed_pred), np.array(self.driving_dir_pred),
                                               self.args)
            self.train_epoch_results['driving_results'] = driving_results
            # {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}


        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename='train')

        # write scalar to tensorboard
        if writer:
            for key in ['speed_Acc', 'speed_mAcc', 'direction_Acc', 'direction_mAcc']: # driving_results.keys(): #
                if key not in driving_results.keys():
                    continue
                val = driving_results[key]
                print("results: ", key, val)
                writer.add_scalar(f'Train/Results/{key}', val, self.epoch)
        print('----------------------------------------------------------- ')


    def eval_epoch_reset(self, epoch, nitern, args=None):
        # 1. initialize log info
        self.frames_list = []
        self.video_list = []
        # (1.1) loss log list
        self.log_loss_total = AverageMeter()
        self.log_loss_driving_speed = AverageMeter()
        self.log_loss_driving_dir = AverageMeter()
        # (1.2) driving
        self.driving_speed_gt = []
        self.driving_speed_pred = []
        self.driving_dir_gt = []
        self.driving_dir_pred = []
        # (1.3) store all results
        self.eval_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

    def eval_driving_batch_update(self, itern, data, speed_gt, direction_gt, speed_pred_logit, dir_pred_logit,
                                 reason_gt=None, reason_pred=None):
        # 3. Update training info
        # (3.1) loss log list
        bs = speed_gt.shape[0]
        # self.frames_list.extend(data['frames'].detach().cpu().numpy())  # bs x sq_length(60)
        # assert len(self.frames_list[0]) == self.args.observe_length
        # self.video_list.extend(data['video_id'])  # bs
        # (3.2) training data info

        self.driving_speed_gt.extend(speed_gt)  # bs
        self.driving_dir_gt.extend(direction_gt)
        self.driving_speed_pred.extend(np.argmax(speed_pred_logit, axis=-1))  # bs
        self.driving_dir_pred.extend(np.argmax(dir_pred_logit, axis=-1))  # bs
        if reason_pred is not None:
            pass # store reason prediction


    def eval_driving_epoch_calculate(self, writer):
        print('----------- Evaluate results: ------------------------------------ ')
        if self.driving:
            driving_results = evaluate_driving(np.array(self.driving_speed_gt), np.array(self.driving_dir_gt),
                                               np.array(self.driving_speed_pred), np.array(self.driving_dir_pred),
                                               self.args)
            self.eval_epoch_results['driving_results'] = driving_results
            # {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}
            for key in self.eval_epoch_results['driving_results'].keys():
                print(key, self.eval_epoch_results['driving_results'][key])
        # Update epoch to all results
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename='eval')


        # write scalar to tensorboard
        if writer:
            for key in ['speed_Acc', 'speed_mAcc', 'direction_Acc', 'direction_mAcc']:
                if key not in driving_results.keys():
                    continue
                val = driving_results[key]
                print("results: ", key, val)
                writer.add_scalar(f'Eval/Results/{key}', val, self.epoch)
        print('log info finished')
        print('----------------------finished results calculation------------------------------------- ')

    def log_args(self, args):
        args_file = os.path.join(self.args.checkpoint_path, 'args.txt')
        with open(args_file, 'a') as f:
            json.dump(args.__dict__, f, indent=2)
        ''' 
            parser = ArgumentParser()
            args = parser.parse_args()
            with open('commandline_args.txt', 'r') as f:
            args.__dict__ = json.load(f)
        '''


    def log_msg(self, msg: str, filename: str = None):
        if not filename:
            filename = os.path.join(self.args.checkpoint_path, 'log.txt')
        else:
            pass
        savet_to_file = filename
        with open(savet_to_file, 'a') as f:
            f.write(str(msg) + '\n')

    def log_info(self, epoch: int, info: dict, filename: str = None):
        if not filename:
            filename = 'log.txt'
        else:
            pass
        for key in info:
            savet_to_file = os.path.join(self.args.checkpoint_path, filename + '_' + key + '.txt')
            self.log_msg(msg='Epoch {} \n --------------------------'.format(epoch), filename=savet_to_file)
            with open(savet_to_file, 'a') as f:
                    if type(info[key]) == str:
                        f.write(info[key] + "\n")
                    elif type(info[key]) == dict:
                        for k in info[key]:
                            f.write(k + ": " + str(info[key][k]) + "\n")
                    else:
                        f.write(str(info[key]) + "\n")
            self.log_msg(msg='.................................................'.format(self.epoch), filename=savet_to_file)
