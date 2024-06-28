import os
from typing import Any

import numpy as np
from torch import nn

from data.custom_dataset import T_drivingBatch, T_drivingSample, T_intentBatch
from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from opts import get_opts
from utils.args import DefaultArguments
from utils.cuda import *
from utils.lr_finder import LRFinder, TrainDataLoaderIter


class TrainIterBbox(TrainDataLoaderIter):
    def __init__(self, ag: DefaultArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = ag

    def inputs_labels_from_batch(self, batch_data):
        return (
            batch_data["bboxes"][:, : self.args.observe_length, :].type(FloatTensor)
            / self.args.image_shape[0]
        ), batch_data["bboxes"][:, self.args.observe_length :, :].type(
            FloatTensor
        ) / self.args.image_shape[
            0
        ]


class TrainIterGlobalIntent(TrainDataLoaderIter):
    def __init__(self, ag: DefaultArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = ag

    def inputs_labels_from_batch(
        self,
        batch_data: dict[
            {
                "global_featmaps": list[Any] | torch.Tensor,
                "image": list[Any] | torch.Tensor,
                "bboxes": torch.Tensor,
            }
        ],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out: torch.Tensor = (
            batch_data["bboxes"][:, self.args.observe_length :, :].type(FloatTensor)
            / self.args.image_shape[0]
        )
        bbox: torch.Tensor
        if self.args.freeze_backbone:
            visual_embeddings: torch.Tensor = batch_data["global_featmaps"]
            bbox = (
                batch_data["bboxes"][:, : self.args.observe_length, :].type(FloatTensor)
                / self.args.image_shape[0]
            )
            return (visual_embeddings, bbox), out

        images: torch.Tensor = batch_data["image"][
            :, : self.args.observe_length, :, :, :
        ].type(FloatTensor)
        bbox = (
            batch_data["bboxes"][:, : self.args.observe_length, :].type(FloatTensor)
            / self.args.image_shape[0]
        )

        return (images, bbox), out


class TrainIterGlobalDecision(TrainDataLoaderIter):
    def __init__(self, ag: DefaultArguments, *args, **kwargs):  # type: ignore[reportMissingParameterType]
        super().__init__(*args, **kwargs)
        self.args = ag

    def inputs_labels_from_batch(
        self, batch_data: T_drivingBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lbl_speed: torch.Tensor = batch_data["label_speed"].type(LongTensor)
        lbl_dir: torch.Tensor = batch_data["label_direction"].type(LongTensor)
        out = torch.cat([lbl_speed, lbl_dir], dim=0)
        if self.args.freeze_backbone:
            visual_embeddings: torch.Tensor = batch_data["global_featmaps"]
            return visual_embeddings, out

        images: torch.Tensor = batch_data["image"][
            :, : self.args.observe_length, :, :, :
        ].type(FloatTensor)
        return images, out


class TrainIterTransformerTraj(TrainDataLoaderIter):
    def __init__(self, ag: DefaultArguments, *args, **kwargs):  # type: ignore[reportMissingParameterType]
        super().__init__(*args, **kwargs)
        self.args = ag

    def inputs_labels_from_batch(
        self, batch_data: T_intentBatch
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        past_future_values: torch.Tensor = batch_data["bboxes"]
        past_future_feats: torch.Tensor = batch_data["total_frames"]

        inputs = (past_future_values, past_future_feats)
        targets: torch.Tensor = batch_data["bboxes"][
            :, self.args.observe_length :, :
        ].type(FloatTensor)
        return inputs, targets


class TrainIterTransformerTrajPose(TrainDataLoaderIter):
    def __init__(self, ag: DefaultArguments, *args, **kwargs):  # type: ignore[reportMissingParameterType]
        super().__init__(*args, **kwargs)
        self.args = ag

    def inputs_labels_from_batch(
        self, batch_data: T_intentBatch
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        past_future_values: torch.Tensor = batch_data["bboxes"]
        past_future_feats: torch.Tensor = batch_data["total_frames"]
        past_future_pose: torch.Tensor = batch_data["pose"]

        inputs = (past_future_values, past_future_feats, past_future_pose)
        targets: torch.Tensor = batch_data["bboxes"][
            :, self.args.observe_length :, :
        ].type(FloatTensor)
        return inputs, targets


def main(args: DefaultArguments):
    amp_config = {"device_type": "cuda", "dtype": torch.bfloat16}
    grad_scaler = torch.cuda.amp.GradScaler()

    # 1. Load database
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    train_loader, _, _ = get_dataloader(args)
    args.steps_per_epoch = int(np.ceil(len(train_loader.dataset) / args.batch_size))
    criterion: nn.Module
    if "global" not in args.model_name:
        if "transformer" not in args.model_name:
            train_data_iter = TrainIterBbox(args, train_loader)
            criterion = torch.nn.L1Loss().to(DEVICE)
        else:
            if "pose" in args.model_name:
                train_data_iter = TrainIterTransformerTrajPose(args, train_loader)
                criterion = nn.NLLLoss().to(DEVICE)
            else:
                train_data_iter = TrainIterTransformerTraj(args, train_loader)
                criterion = nn.NLLLoss().to(DEVICE)
    elif "driving_decision" != args.task_name:
        train_data_iter = TrainIterGlobalIntent(args, train_loader)
        criterion = nn.MSELoss().to(DEVICE)
    else:
        train_data_iter = TrainIterGlobalDecision(args, train_loader)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    # 2. Create model
    model, _, _ = build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    model = nn.DataParallel(model)
    # 3. Find LR.
    lr_finder = LRFinder(
        model,
        optimizer,
        criterion,
        device="cuda",
        amp_backend="torch",
        amp_config=amp_config,
        grad_scaler=grad_scaler,
    )
    lr_finder.range_test(train_data_iter, start_lr=1e-6, end_lr=0.1, num_iter=300)
    lr_finder.plot()


if __name__ == "__main__":
    args = get_opts()
    args.task_name = args.task_name if args.task_name else "ped_traj"
    args.persist_dataloader = True

    match args.task_name:
        case "ped_intent":
            args.database_file = "intent_database_train.pkl"
            args.intent_model = True
            args.batch_size = 256
            # intent prediction
            args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
            args.intent_type = "mean"  # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
            args.intent_loss = ["bce"]
            args.intent_disagreement = (
                1  # -1: not use disagreement 1: use disagreement to reweigh samples
            )
            args.intent_positive_weight = (
                0.5  # Reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)
            )
            args.predict_length = 1  # only make one intent prediction
            args.model_name = "tcn_int_bbox"
            args.loss_weights = {
                "loss_intent": 1.0,
                "loss_traj": 0.0,
                "loss_driving": 0.0,
            }
            args.load_image = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE

        case "ped_traj":
            args.database_file = "traj_database_train.pkl"
            args.intent_model = False  # if (or not) use intent prediction module to support trajectory prediction
            args.traj_model = True
            # args.traj_loss = ["bbox_huber"]
            args.traj_loss = ["bbox_l2"]
            # args.batch_size = [256]
            # args.batch_size = 64
            args.batch_size = 256
            args.predict_length = 45
            # args.model_name = "tcn_traj_bbox"
            # args.model_name = "tcn_traj_bbox_int"
            # args.model_name = "tcn_traj_global"
            # args.model_name = "tcan_traj_bbox"
            # args.model_name = "tcan_traj_bbox_int"
            # args.model_name = "tcan_traj_global"
            args.model_name = "transformer_traj_bbox_pose"
            args.loss_weights = {
                "loss_intent": 0.0,
                "loss_traj": 1.0,
                "loss_driving": 0.0,
            }
            args.load_image = False
            args.backbone = ""
            args.freeze_backbone = False
            # args.load_image = True
            # args.backbone = "resnet50"
            # args.freeze_backbone = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
            args.n_layers = 4
            args.kernel_size = 2

        case "driving_decision":
            args.database_file = "driving_database_train.pkl"
            args.driving_loss = ["cross_entropy"]
            args.batch_size = 256
            args.model_name = "restcn_driving_global"
            # args.model_name = "reslstm_driving_global"
            args.loss_weights = {
                "loss_intent": 0.0,
                "loss_traj": 0.0,
                "loss_driving": 1.0,
            }
            # args.load_image = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
            args.load_image = False
            args.backbone = "resnet50"
            args.freeze_backbone = True

    args.observe_length = 15
    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = "enlarge"
    args.normalize_bbox = None

    main(args)
