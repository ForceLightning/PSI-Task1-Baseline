"""Builds the model, optimizer, and scheduler objects for training.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import TimeSeriesTransformerConfig

from models.driving_modules.model_lstm_driving_global import ResLSTMDrivingGlobal
from models.driving_modules.model_tcn_driving_global import ResTCNDrivingGlobal
from models.intent_modules.model_lstm_int_bbox import LSTMIntBbox
from models.intent_modules.model_tcn_int_bbox import TCNINTBbox
from models.traj_modules.model_lstmed_traj_bbox import LSTMedTrajBbox
from models.traj_modules.model_tcan_traj_bbox import TCANTrajBbox, TCANTrajBboxInt
from models.traj_modules.model_tcn_traj_bbox import TCNTrajBbox, TCNTrajBboxInt
from models.traj_modules.model_tcn_traj_global import TCANTrajGlobal, TCNTrajGlobal
from models.traj_modules.model_transformer_traj_bbox import (
    TransformerTrajBbox,
    TransformerTrajBboxPose,
    TransformerTrajIntentBboxPose,
)
from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *

TRANSFORMER_BASE_CONFIG = {
    "d_model": 256,
    "encoder_attention_heads": 4,
    "decoder_attention_heads": 4,
    "encoder_ffn_dim": 128,
    "decoder_ffn_dim": 128,
    "dropout": 0.125,
    "attention_dropout": 0.125,
    "lags_sequence": [1],  # defaults to 0..7
    "activation_function": "mish",
    "optimizer": "AdamW",
    "opt_wd": 0.1,
}


def build_model(
    args: DefaultArguments,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Builds the model, optimizer, and scheduler objects for training.

    :param DefaultArguments args: Training arguments.
    :returns: Model, Optimizer, and Scheduler in a tuple.
    :rtype: tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
    """
    match args.model_name:
        case "lstm_int_bbox":
            model = get_lstm_intent_bbox(args).to(DEVICE)
        case "tcn_int_bbox":
            model = get_tcn_intent_bbox(args).to(DEVICE)
        case "lstm_traj_bbox" | "lstmed_traj_bbox":
            model = get_lstmed_traj_bbox(args).to(DEVICE)
        case "tcn_traj_bbox":
            model = get_tcn_traj_bbox(args).to(DEVICE)
        case "tcn_traj_bbox_int":
            model = get_tcn_traj_bbox_int(args).to(DEVICE)
        case "tcn_traj_global":
            model = get_tcn_traj_bbox_global(args).to(DEVICE)
        case "tcan_traj_bbox":
            model = get_tcan_traj_bbox(args).to(DEVICE)
        case "transformer_traj_bbox":
            model = get_transformer_traj_bbox(args).to(DEVICE)
        case "transformer_traj_bbox_pose":
            model = get_transformer_traj_bbox_pose(args).to(DEVICE)
        case "transformer_traj_intent_bbox_pose":
            model = get_transformer_traj_intent_bbox_pose(args).to(DEVICE)
        case "tcan_traj_global":
            model = get_tcan_traj_bbox_global(args).to(DEVICE)
        case "reslstm_driving_global":
            model = get_lstm_driving_global(args).to(DEVICE)
        case "restcn_driving_global":
            model = get_tcn_driving_global(args).to(DEVICE)
        case _:
            raise ValueError(f"Model {args.model_name} not implemented yet.")

    optimizer, scheduler = model.build_optimizer(args)
    return model, optimizer, scheduler


# 1. Intent prediction
# 1.1 input bboxes only
def get_lstm_intent_bbox(args: DefaultArguments) -> LSTMIntBbox:
    """Gets the LSTM model for intent prediction.
    :param DefaultArguments args: Training arguments.
    """
    model_configs: dict[str, ModelOpts] = {}
    model_configs["intent_model_opts"] = {
        "enc_in_dim": 4,  # input bbox (normalized OR not) + img_context_feat_dim
        "enc_out_dim": 64,
        "dec_in_emb_dim": None,  # encoder output + bbox
        "dec_out_dim": 64,
        "output_dim": 1,  # intent prediction, output logits, add activation later
        "n_layers": 1,
        "dropout": 0.5,
        "observe_length": args.observe_length,  # 15
        "predict_length": 1,  # only predict one intent
        "return_sequence": False,  # False for reason/intent/trust. True for trajectory
        "output_activation": "None",  # [None | tanh | sigmoid | softmax]
    }
    args.model_configs = model_configs
    model = LSTMIntBbox(args, model_configs["intent_model_opts"])
    return model


def get_tcn_intent_bbox(args: DefaultArguments) -> TCNINTBbox:
    """Gets the TCN model for intent prediction.
    :param DefaultArguments args: Training arguments.
    """
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
        "output_activation": "None",  # [None | tanh | sigmoid | softmax]
        "use_skip_connection": True,
    }
    args.model_configs = model_configs
    model = TCNINTBbox(args, model_configs["intent_model_opts"])
    return model


def get_lstmed_traj_bbox(args: DefaultArguments) -> LSTMedTrajBbox:
    """Gets the LSTM model for trajectory prediction.
    :param DefaultArguments args: Training arguments.
    """
    model_configs = {}
    model_configs["traj_model_opts"] = {
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
    model = LSTMedTrajBbox(args, model_configs["traj_model_opts"])
    return model


def get_tcn_traj_bbox(args: DefaultArguments) -> TCNTrajBbox:
    """Gets the TCN model for trajectory prediction.
    :param DefaultArguments args: Training arguments.
    """
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


def get_tcn_traj_bbox_int(args: DefaultArguments) -> TCNTrajBboxInt:
    """Gets the TCN model for trajectory prediction with intent.
    :param DefaultArguments args: Training arguments.
    """
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


def get_tcan_traj_bbox(args: DefaultArguments) -> TCANTrajBbox:
    """Gets the TCAN model for trajectory prediction.
    :param DefaultArguments args: Training arguments.
    """
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
    model = TCANTrajBbox(args, model_configs["traj_model_opts"])
    return model


def get_tcan_traj_bbox_int(args: DefaultArguments) -> TCANTrajBboxInt:
    """Gets the TCAN model for trajectory prediction with intent.
    :param DefaultArguments args: Training arguments.
    """
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
    model = TCANTrajBboxInt(args, model_configs["traj_model_opts"])
    return model


def get_transformer_traj_bbox(args: DefaultArguments) -> TransformerTrajBbox:
    """Gets the Transformer model for trajectory prediction.
    :param DefaultArguments args: Training arguments.
    """
    model_configs: dict[str, Any] = {
        "prediction_length": args.predict_length,
        "context_length": args.observe_length - 1,
        "input_size": 4,  # number of bbox coords per frame
        "num_time_features": 1,  # number of additional features
        "encoder_layers": args.n_layers,
        "decoder_layers": args.n_layers,
    }
    for key, val in TRANSFORMER_BASE_CONFIG.items():
        if model_configs.get(key) is None:
            model_configs[key] = val
    args.model_configs = model_configs
    config = TimeSeriesTransformerConfig(**model_configs)  # type: ignore[reportArgumentType]
    model = TransformerTrajBbox(args, config)
    return model


def get_transformer_traj_bbox_pose(args: DefaultArguments) -> TransformerTrajBboxPose:
    """Gets the Transformer model for trajectory prediction with pose.
    :param DefaultArguments args: Training arguments.
    """
    model_configs: dict[str, Any] = {
        "prediction_length": args.predict_length,
        "context_length": args.observe_length - 1,
        "input_size": 4,  # number of bbox coords per frame
        "num_time_features": 1,  # number of additional monotonically increasing features
        "num_dynamic_real_features": 34,  # number of dynamic features (pose data)
        "encoder_layers": args.n_layers,
        "decoder_layers": args.n_layers,
    }
    for key, val in TRANSFORMER_BASE_CONFIG.items():
        if model_configs.get(key) is None:
            model_configs[key] = val
    args.model_configs = model_configs
    config = TimeSeriesTransformerConfig(**model_configs)  # type: ignore[reportArgumentType]
    model = TransformerTrajBboxPose(args, config)
    return model


def get_transformer_traj_intent_bbox_pose(
    args: DefaultArguments,
) -> TransformerTrajIntentBboxPose:
    """Gets the Transformer model for trajectory and intent prediction with bbox and
    pose data.
    :param DefaultArguments args: Training arguments.
    """
    model_configs: dict[str, Any] = {
        "prediction_length": args.predict_length,
        "context_length": args.observe_length - 1,
        "input_size": 4,  # number of bbox coords per frame
        "num_time_features": 1,  # number of additional monotonically increasing features
        "num_dynamic_real_features": 34,  # number of dynamic features (pose data)
        "encoder_layers": args.n_layers,
        "decoder_layers": args.n_layers,
    }
    for key, val in TRANSFORMER_BASE_CONFIG.items():
        if model_configs.get(key) is None:
            model_configs[key] = val
    args.model_configs = model_configs
    config = TimeSeriesTransformerConfig(**model_configs)  # type: ignore[reportArgumentType]
    model = TransformerTrajIntentBboxPose(args, config)
    return model


def get_tcn_traj_bbox_global(args: DefaultArguments) -> TCNTrajGlobal:
    """Gets the TCN model for trajectory prediction with global image features.
    :param DefaultArguments args: Training arguments.
    """
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


def get_tcan_traj_bbox_global(args: DefaultArguments) -> TCANTrajGlobal:
    """Gets the TCAN model for trajectory prediction with global image features.
    :param DefaultArguments args: Training arguments.
    """
    model_configs = {}
    model_configs["traj_model_opts"] = {
        "enc_in_dim": 516,
        "enc_out_dim": 64,
        "dec_in_emb_dim": 0,
        "dec_out_dim": 128,
        "output_dim": 4,
        "n_layers": args.n_layers,
        "dropout": 0.125,
        "kernel_size": args.kernel_size,
        "observe_length": args.observe_length,
        "predict_length": args.predict_length,
        "return_sequence": True,
        "output_activation": "None",
        "num_heads": 4,
    }
    args.model_configs = model_configs
    model = TCANTrajGlobal(args, model_configs["traj_model_opts"])
    return model


# 3. driving decision prediction
# 3.1 input global images only
def get_lstm_driving_global(args: DefaultArguments):
    """Gets the LSTM model for driving decision prediction with global image features.
    :param DefaultArguments args: Training arguments.
    """
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


def get_tcn_driving_global(args: DefaultArguments):
    """Gets the TCN model for driving decision prediction with global image features.
    :param DefaultArguments args: Training arguments.
    """
    model_configs = {}
    model_configs["driving_model_opts"] = {
        "enc_in_dim": 0,  # input bbox (normalized OR not) + img_context_feat_dim
        "enc_out_dim": 64,
        "dec_in_emb_dim": 0,  # intent(1), speed(1), rsn(? Bert feats dim)
        "dec_out_dim": 64,
        "output_dim": 3,  # 3 classes each.
        "n_layers": args.n_layers,
        "dropout": 0.125,
        "kernel_size": args.kernel_size,
        "observe_length": args.observe_length,
        "predict_length": args.predict_length,
        "return_sequence": True,
        "output_activation": "None",
        "opt_name": "AdamW",
    }
    args.model_configs = model_configs
    model = ResTCNDrivingGlobal(args, model_configs["driving_model_opts"])
    return model


if __name__ == "__main__":
    from torchinfo import summary

    args = DefaultArguments()
    args.n_layers = 8
    args.kernel_size = 2
    args.observe_length = 15
    args.predict_length = 45
    args.freeze_backbone = True
    args.backbone = "resnet50"
    args.load_image = True
    model_configs = {}
    model_configs["traj_model_opts"] = {
        "enc_in_dim": 516,
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
        "num_heads": 4,
    }
    args.model_configs = model_configs
    args.batch_size = 256
    model = TCANTrajGlobal(args, model_configs["traj_model_opts"]).to(DEVICE)

    # Get summary of each segment
    # input_size_1 = (args.batch_size, 15, 2048)
    # print(summary(model.cnn_encoder, input_size_1))
    #
    # input_size_2 = (args.batch_size, args.observe_length, 516)
    # print(summary(model.tcn, input_size_2))

    # input_size = (args.batch_size, args.observe_length, 516)
    # submodel = nn.Sequential(*[x for _, x in list(model.named_children())[1:]])
    # print(summary(submodel, input_size))

    data = [
        {
            "global_featmaps": torch.rand(
                (args.batch_size, args.observe_length, 2048)
            ).to(DEVICE),
            "bboxes": torch.rand((args.batch_size, args.observe_length, 4)).to(DEVICE),
        }
    ]
    out: torch.Tensor = model(data[0])
    print(out.shape)
    print(out)
    print("\n\n")
    print(summary(model, input_data=data, batch_dim=0))
