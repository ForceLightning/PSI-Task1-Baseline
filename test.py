import json
import os
from typing import Any

import numpy as np
from numpy import typing as npt
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from utils.args import DefaultArguments
from utils.cuda import *
from utils.log import RecordResults


def validate_intent(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
):
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)
        # intent_pred: logit output, bs
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == "mean" and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent = data["intention_binary"][:, args.observe_length].type(
                FloatTensor
            )
            gt_intent_prob = data["intention_prob"][:, args.observe_length].type(
                FloatTensor
            )
            # gt_disagreement = data['disagree_score'][:, args.observe_length]
            # gt_consensus = (1 - gt_disagreement).to(device)

        recorder.eval_intent_batch_update(
            itern,
            data,
            gt_intent.detach().cpu().numpy(),
            intent_prob.detach().cpu().numpy(),
            gt_intent_prob.detach().cpu().numpy(),
        )

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

    recorder.eval_intent_epoch_calculate(writer)
    return recorder


def test_intent(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
) -> RecordResults:
    model.eval()
    niters = len(dataloader)
    recorder.eval_epoch_reset(epoch, niters)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)
        # intent_pred: logit output, bs x 1
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == "mean" and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent: torch.Tensor = data["intention_binary"][
                :, args.observe_length
            ].type(FloatTensor)
            gt_intent_prob: torch.Tensor = data["intention_prob"][
                :, args.observe_length
            ].type(FloatTensor)

        recorder.eval_intent_batch_update(
            itern,
            data,
            gt_intent.detach().cpu().numpy(),
            intent_prob.detach().cpu().numpy(),
            gt_intent_prob.detach().cpu().numpy(),
        )

    recorder.eval_intent_epoch_calculate(writer)

    return recorder


def predict_intent(
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    dset: str = "test",
) -> None:
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        intent_prob = torch.sigmoid(intent_logit)

        for i in range(len(data["frames"])):
            vid = data["video_id"][i]  # str list, bs x 60
            pid = data["ped_id"][i]  # str list, bs x 60
            fid = (
                data["frames"][i][-1] + 1
            ).item()  # int list, bs x 15, observe 0~14, predict 15th intent
            # gt_int = data['intention_binary'][i][args.observe_length].item()  # int list, bs x 60
            # gt_int_prob = data['intention_prob'][i][args.observe_length].item()  # float list, bs x 60
            # gt_disgr = data['disagree_score'][i][args.observe_length].item()  # float list, bs x 60
            int_prob = intent_prob[i].item()
            int_pred = round(int_prob)  # <0.5 --> 0, >=0.5 --> 1.

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]["intent"] = int_pred
            dt[vid][pid][fid]["intent_prob"] = int_prob

    with open(
        os.path.join(args.checkpoint_path, "results", f"{dset}_intent_pred"), "w"
    ) as f:
        json.dump(dt, f)


def validate_traj(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
    criterion: nn.Module,
) -> float:
    _ = model.eval()
    niters: int = len(dataloader)
    num_samples: int = len(dataloader.dataset)
    val_losses: list[float] = []
    for itern, data in enumerate(dataloader):
        curr_batch_size: int = data["bboxes"].shape[0]
        traj_pred: torch.Tensor = model(data)
        traj_gt: torch.Tensor = data["bboxes"][:, args.observe_length :, :].type(
            FloatTensor
        )
        val_loss: float = (
            criterion(
                traj_pred / args.image_shape[0], traj_gt / args.image_shape[0]
            ).item()
            # / 4
            # / args.observe_length
        )
        val_losses.append(val_loss)
        # bs, ts, _ = traj_gt.shape
        # if args.normalize_bbox == 'subtract_first_frame':
        #     traj_pred = traj_pred + data['bboxes'][:, :1, :].type(FloatTensor)
        recorder.eval_traj_batch_update(
            itern,
            data,
            traj_gt.detach().cpu().numpy(),
            traj_pred.detach().cpu().numpy(),
        )

        if itern % args.print_freq == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} | Val Loss: {val_loss:.4f}"
            )

    val_losses: float = np.mean(val_losses)
    print(f"Epoch {epoch}/{args.epochs} Val Loss: {val_losses:.4f}")
    writer.add_scalar("Losses/val_loss", val_losses, epoch)
    recorder.eval_traj_epoch_calculate(writer)

    return val_losses


def predict_traj(
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    dset: str = "test",
) -> None:
    _ = model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        traj_pred: torch.Tensor = model(data)
        # traj_gt = data['original_bboxes'][:, args.observe_length:, :].type(FloatTensor)
        traj_gt: torch.Tensor = data["bboxes"][:, args.observe_length :, :].type(
            FloatTensor
        )
        # bs, ts, _ = traj_gt.shape
        # print("Prediction: ", traj_pred.shape)

        for i in range(len(data["frames"])):  # for each sample in a batch
            vid: str = data["video_id"][i]  # str list, bs x 60
            pid: str = data["ped_id"][i]  # str list, bs x 60
            fid: int = (
                data["frames"][i][-1] + 1
            ).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]["traj"] = traj_pred[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    # print("saving prediction...")
    with open(
        os.path.join(args.checkpoint_path, "results", f"{dset}_traj_pred.json"), "w"
    ) as f:
        json.dump(dt, f)


def get_test_traj_gt(
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    dset: str = "test",
) -> None:
    _ = model.eval()
    gt = {}
    for data in dataloader:
        # traj_pred: torch.Tensor = model(data)
        traj_gt: torch.Tensor = data["bboxes"][:, args.observe_length :, :].type(
            FloatTensor
        )
        # traj_gt = data['original_bboxes'][:, args.observe_length:, :].type(FloatTensor)
        # bs, ts, _ = traj_gt.shape
        # print("Prediction: ", traj_pred.shape)

        for i in range(len(data["frames"])):  # for each sample in a batch
            vid: str = data["video_id"][i]  # str list, bs x 60
            pid: str = data["ped_id"][i]  # str list, bs x 60
            fid: int = (
                data["frames"][i][-1] + 1
            ).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in gt:
                gt[vid] = {}
            if pid not in gt[vid]:
                gt[vid][pid] = {}
            if fid not in gt[vid][pid]:
                gt[vid][pid][fid] = {}
            gt[vid][pid][fid]["traj"] = traj_gt[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    with open(os.path.join(f"./test_gt/{dset}_traj_gt.json"), "w") as f:
        json.dump(gt, f)


@torch.no_grad()
def validate_driving(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
    criterion: nn.Module,
):
    print(f"Validate ...")
    model.eval()
    niters = len(dataloader)
    val_losses: list[float] = []
    for itern, data in enumerate(dataloader):
        pred_speed_logit, pred_dir_logit = model(data)
        lbl_speed = data["label_speed"].type(LongTensor)  # bs x 1
        lbl_dir = data["label_direction"].type(LongTensor)  # bs x 1
        val_loss: float = (
            criterion(pred_speed_logit, lbl_speed).item()
            + criterion(pred_dir_logit, lbl_dir).item()
        )
        val_losses.append(val_loss)
        recorder.eval_driving_batch_update(
            itern,
            data,
            lbl_speed.detach().cpu().numpy(),
            lbl_dir.detach().cpu().numpy(),
            pred_speed_logit.detach().cpu().numpy(),
            pred_dir_logit.detach().cpu().numpy(),
        )

        if itern % args.print_freq == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} | Val Loss: {val_loss:.4f}"
            )

        del data
        del pred_speed_logit
        del pred_dir_logit

    val_losses: float = np.mean(val_losses)
    print(f"Epoch {epoch}/{args.epochs} | Val Loss: {val_losses:.4f}")
    writer.add_scalar("Losses/val_loss", val_losses, epoch)
    recorder.eval_driving_epoch_calculate(writer)

    return val_losses


@torch.no_grad()
def predict_driving(model, dataloader, args, dset="test"):
    print(f"Predict and save prediction of {dset} set...")
    model.eval()
    dt = {}
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        pred_speed_logit, pred_dir_logit = model(data)
        # lbl_speed = data['label_speed']  # bs x 1
        # lbl_dir = data['label_direction']  # bs x 1
        # print("batch size: ", len(data['frames']), len(data['video_id']))
        for i in range(len(data["frames"])):  # for each sample in a batch
            # print(data['video_id'])
            vid = data["video_id"][0][i]  # str list, bs x 60
            fid = (
                data["frames"][i][-1] + 1
            ).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if fid not in dt[vid]:
                dt[vid][fid] = {}
            dt[vid][fid]["speed"] = torch.argmax(pred_speed_logit[i]).item()
            dt[vid][fid]["direction"] = torch.argmax(pred_dir_logit[i]).item()

        if itern % args.print_freq == 10:
            print(f"Predicting driving decision of Batch {itern}/{niters}")
        del data
        del pred_speed_logit
        del pred_dir_logit

    print("Saving prediction to file...")
    with open(
        os.path.join(args.checkpoint_path, "results", f"{dset}_driving_pred.json"), "w"
    ) as f:
        json.dump(dt, f)
