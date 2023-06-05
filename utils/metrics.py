from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
import torch.nn.functional as F

def evaluate_driving(target_speed, target_dir, pred_speed, pred_dir, args):
    results = {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}
    print("Evaluating Driving Decision Prediction ...")

    bs = target_speed.shape[0]
    results['speed_Acc'] = accuracy_score(target_speed, pred_speed)
    results['direction_Acc'] = accuracy_score(target_dir, pred_dir)

    speed_matrix = confusion_matrix(target_speed, pred_speed)
    results['speed_confusion_matrix'] = speed_matrix
    sum_cnt = speed_matrix.sum(axis=1)
    sum_cnt = np.array([max(1, i) for i in sum_cnt])
    speed_cls_wise_acc = speed_matrix.diagonal() / sum_cnt
    results['speed_mAcc'] = np.mean(speed_cls_wise_acc)

    dir_matrix = confusion_matrix(target_dir, pred_dir)
    results['dir_confusion_matrix'] = dir_matrix
    sum_cnt = dir_matrix.sum(axis=1)
    sum_cnt = np.array([max(1, i) for i in sum_cnt])
    dir_cls_wise_acc = dir_matrix.diagonal() / sum_cnt
    results['dir_mAcc'] = np.mean(dir_cls_wise_acc)


    return results

def shannon(data):
    shannon = -np.sum(data*np.log2(data))
    return shannon
