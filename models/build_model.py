import os

import numpy as np
import torch
from models.driving_modules.model_lstm_driving_global import ResLSTMDrivingGlobal
from models.intent_modules.model_tcn_int_bbox import TCNINTBbox
from models.traj_modules.model_tcn_traj_bbox import (
    TCNTrajBbox,
    TCNTrajBboxInt,
)
from models.traj_modules.model_tcn_traj_semantic import TCNTrajGlobal

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def build_model(args):
    # Intent models
    match args.model_name:
        case "tcn_int_bbox":
            model = get_tcn_intent_bbox(args).to(device)
        case "tcn_traj_bbox":
            model = get_tcn_traj_bbox(args).to(device)
        case "tcn_traj_bbox_int":
            model = get_tcn_traj_bbox_int(args).to(device)
        case "tcn_traj_global":
            model = get_tcn_traj_bbox_global(args).to(device)
        case "reslstm_driving_global":
            model = get_lstm_driving_global(args).to(device)
        case _:
            raise ValueError(f"Model {args.model_name} not implemented yet.")

    optimizer, scheduler = model.build_optimizer(args)
    return model, optimizer, scheduler


# 1. Intent prediction
# 1.1 input bboxes only
def get_tcn_intent_bbox(args):
    model_configs = {}
    model_configs["intent_model_opts"] = {
        "enc_in_dim": 4,  # input bbox (normalized OR not) + img_context_feat_dim
        "enc_out_dim": 64,
        "dec_in_emb_dim": None,  # encoder output + bbox
        "dec_out_dim": 64,
        "output_dim": 1,  # intent prediction, output logits, add activation later
        "n_layers": args.n_layers,
        "dropout": 0.125,
        "kernel_size": args.kernel_size,
        "observe_length": args.observe_length,  # 15
        "predict_length": 1,  # only predict one intent
        "return_sequence": False,  # False for reason/intent/trust. True for trajectory
        "output_activation": "None",
        "use_skip_connection": True,  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs
    model = TCNINTBbox(args, model_configs)
    return model


def get_tcn_traj_bbox(args):
    model_configs = {}
    model_configs["traj_model_opts"] = {
        "enc_in_dim": 4,
        "enc_out_dim": 64,
        "dec_in_emb_dim": 0,
        "dec_out_dim": 64,
        "output_dim": 4,
        "n_layers": args.n_layers,
        "dropout": 0.125,
        "kernel_size": args.kernel_size,
        "observe_length": args.observe_length,
        "predict_length": args.predict_length,
        "return_sequence": True,
        "output_activation": "None",
    }
    args.model_configs = model_configs
    model = TCNTrajBbox(args, model_configs["traj_model_opts"])
    return model


def get_tcn_traj_bbox_int(args):
    model_configs = {}
    model_configs["traj_model_opts"] = {
        "enc_in_dim": 5,
        "enc_out_dim": 64,
        "dec_in_emb_dim": 0,
        "dec_out_dim": 64,
        "output_dim": 4,
        "n_layers": args.n_layers,
        "dropout": 0.125,
        "kernel_size": args.kernel_size,
        "observe_length": args.observe_length,
        "predict_length": args.predict_length,
        "return_sequence": True,
        "output_activation": "None",
    }
    args.model_configs = model_configs
    model = TCNTrajBboxInt(args, model_configs["traj_model_opts"])
    return model


def get_tcn_traj_bbox_global(args):
    model_configs = {}
    model_configs["traj_model_opts"] = {
        "enc_in_dim": 4,
        "enc_out_dim": 64,
        "dec_in_emb_dim": 0,
        "dec_out_dim": 64,
        "output_dim": 4,
        "n_layers": args.n_layers,
        "dropout": 0.125,
        "kernel_size": args.kernel_size,
        "observe_length": args.observe_length,
        "predict_length": args.predict_length,
        "return_sequence": True,
        "output_activation": "None",
    }
    args.model_configs = model_configs
    model = TCNTrajGlobal(args, model_configs["traj_model_opts"])
    return model


# 3. driving decision prediction
# 3.1 input global images only
def get_lstm_driving_global(args):
    model_configs = {}
    model_configs["driving_model_opts"] = {
        "enc_in_dim": 4,  # input bbox (normalized OR not) + img_context_feat_dim
        "enc_out_dim": 64,
        "dec_in_emb_dim": 0,  # intent(1), speed(1), rsn(? Bert feats dim)
        "dec_out_dim": 64,
        "output_dim": 4,  # intent prediction, output logits, add activation later
        "n_layers": 1,
        "dropout": 0.5,
        "observe_length": args.observe_length,  # 15
        "predict_length": args.predict_length,  # only predict one intent
        "return_sequence": True,  # False for reason/intent/trust. True for trajectory
        "output_activation": "None",  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs
    model = ResLSTMDrivingGlobal(args, model_configs["driving_model_opts"])
    return model
