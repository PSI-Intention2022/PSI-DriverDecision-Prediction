import numpy as np
import torch
import os
from .intent_modules.model_lstm_int_bbox import LSTMIntBbox
from .traj_modules.model_lstmed_traj_bbox import LSTMedTrajBbox
from .driving_modules.model_lstm_driving_global import ResLSTMDrivingGlobal

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def build_model(args):
    # Intent models
    if args.model_name == 'lstm_int_bbox':
        model = get_lstm_intent_bbox(args).to(device)
        optimizer, scheduler = model.build_optimizer(args)
        return model, optimizer, scheduler
    elif args.model_name == 'lstmed_traj_bbox':
        model = get_lstmed_traj_bbox(args).to(device)
        optimizer, scheduler = model.build_optimizer(args)
        return model, optimizer, scheduler
    elif args.model_name == 'reslstm_driving_global':
        model = get_lstm_driving_global(args).to(device)
        optimizer, scheduler = model.build_optimizer(args)
        return model, optimizer, scheduler

# 1. Intent prediction
# 1.1 input bboxes only
def get_lstm_intent_bbox(args):
    model_configs = {}
    model_configs['intent_model_opts'] = {
        'enc_in_dim': 4,  # input bbox (normalized OR not) + img_context_feat_dim
        'enc_out_dim': 64,
        'dec_in_emb_dim': None,  # encoder output + bbox
        'dec_out_dim': 64,
        'output_dim': 1,  # intent prediction, output logits, add activation later
        'n_layers': 1,
        'dropout': 0.5,
        'observe_length': args.observe_length,  # 15
        'predict_length': 1,  # only predict one intent
        'return_sequence': False,  # False for reason/intent/trust. True for trajectory
        'output_activation': 'None'  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs
    model = LSTMIntBbox(args, model_configs)
    return model

# 2. Trajectory prediction
# 2.1 input bboxes only
def get_lstmed_traj_bbox(args):
    model_configs = {}
    model_configs['traj_model_opts'] = {
        'enc_in_dim': 4,  # input bbox (normalized OR not) + img_context_feat_dim
        'enc_out_dim': 64,
        'dec_in_emb_dim': 0,  # intent(1), speed(1), rsn(? Bert feats dim)
        'dec_out_dim': 64,
        'output_dim': 4,  # intent prediction, output logits, add activation later
        'n_layers': 1,
        'dropout': 0.5,
        'observe_length': args.observe_length,  # 15
        'predict_length': args.predict_length,  # only predict one intent
        'return_sequence': True,  # False for reason/intent/trust. True for trajectory
        'output_activation': 'None'  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs
    model = LSTMedTrajBbox(args, model_configs['traj_model_opts'])
    return model



# 3. driving decision prediction
# 3.1 input global images only
def get_lstm_driving_global(args):
    model_configs = {}
    model_configs['driving_model_opts'] = {
        'enc_in_dim': 4,  # input bbox (normalized OR not) + img_context_feat_dim
        'enc_out_dim': 64,
        'dec_in_emb_dim': 0,  # intent(1), speed(1), rsn(? Bert feats dim)
        'dec_out_dim': 64,
        'output_dim': 4,  # intent prediction, output logits, add activation later
        'n_layers': 1,
        'dropout': 0.5,
        'observe_length': args.observe_length,  # 15
        'predict_length': args.predict_length,  # only predict one intent
        'return_sequence': True,  # False for reason/intent/trust. True for trajectory
        'output_activation': 'None'  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs
    model = ResLSTMDrivingGlobal(args, model_configs['driving_model_opts'])
    return model