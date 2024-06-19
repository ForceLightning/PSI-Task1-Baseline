import collections
import gc
import os

import numpy as np
import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from test import validate_driving, validate_intent, validate_traj
from utils.args import DefaultArguments
from utils.log import RecordResults
from utils.resume_training import ResumeTrainingInfo
from utils.cuda import *

# scaler = GradScaler() if CUDA else None
scaler = None


def train_intent(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
) -> None:
    pos_weight = torch.tensor(args.intent_positive_weight).to(
        DEVICE
    )  # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight
        ).to(DEVICE),
        "MSELoss": torch.nn.MSELoss(reduction="none").to(DEVICE),
        "BCELoss": torch.nn.BCELoss().to(DEVICE),
        "CELoss": torch.nn.CrossEntropyLoss(),
    }
    epoch_loss = {"loss_intent": [], "loss_traj": []}
    for epoch in range(1, args.epochs + 1):
        niters = np.ceil(len(train_loader.dataset) / args.batch_size)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_intent_epoch(
            epoch,
            model,
            optimizer,
            criterions,
            epoch_loss,
            train_loader,
            args,
            recorder,
            writer,
        )
        if not isinstance(scheduler, OneCycleLR):
            scheduler.step()

        if epoch % 1 == 0:
            print(
                f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}"
            )

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = np.ceil(len(val_loader.dataset) / args.batch_size)
            recorder.eval_epoch_reset(epoch, niters)
            validate_intent(epoch, model, val_loader, args, recorder, writer)

            # result_path = os.path.join(args.checkpoint_path, 'results', f'epoch_{epoch}')
            # if not os.path.isdir(result_path):
            #     os.makedirs(result_path)
            # recorder.save_results(prefix='')
            # torch.save(model.state_dict(), result_path + f'/state_dict.pth')

        torch.save(model.state_dict(), args.checkpoint_path + f"/latest.pth")


def train_intent_epoch(
    epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer
):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = np.ceil(len(dataloader.dataset) / args.batch_size)
    for itern, data in enumerate(dataloader):
        optimizer.zero_grad()
        intent_logit = model(data)
        # intent_pred: sigmoid output, (0, 1), bs
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        loss_intent: torch.Tensor = torch.tensor(0.0).type(FloatTensor).to(DEVICE)  # type: ignore
        gt_intent: torch.Tensor
        gt_intent_prob: torch.Tensor
        if args.intent_type == "mean" and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent = data["intention_binary"][:, args.observe_length].type(
                FloatTensor
            )
            gt_intent_prob = data["intention_prob"][:, args.observe_length].type(
                FloatTensor
            )

            gt_disagreement = data["disagree_score"][:, args.observe_length]
            gt_consensus = (1 - gt_disagreement).to(DEVICE)

            if "bce" in args.intent_loss:
                loss_intent_bce = criterions["BCEWithLogitsLoss"](
                    intent_logit, gt_intent
                )

                if args.intent_disagreement != -1.0:
                    if args.ignore_uncertain:
                        mask = (gt_consensus > args.intent_disagreement) * gt_consensus
                    else:
                        mask = gt_consensus
                    loss_intent_bce = torch.mean(torch.mul(mask, loss_intent_bce))
                else:  # -1.0, not use reweigh and filter
                    loss_intent_bce = torch.mean(loss_intent_bce)
                batch_losses["loss_intent_bce"].append(loss_intent_bce.item())
                loss_intent += loss_intent_bce

            if "mse" in args.intent_loss:
                loss_intent_mse = criterions["MSELoss"](
                    gt_intent_prob, torch.sigmoid(intent_logit)
                )

                if args.intent_disagreement != -1.0:
                    mask = (gt_consensus > args.intent_disagreement) * gt_consensus
                    loss_intent_mse = torch.mean(torch.mul(mask, loss_intent_mse))
                else:  # -1.0, not use reweigh and filter
                    loss_intent_mse = torch.mean(loss_intent_mse)

                batch_losses["loss_intent_mse"].append(loss_intent_mse.item())

                loss_intent += loss_intent_mse

        loss = args.loss_weights["loss_intent"] * loss_intent.item()

        loss.backward()
        optimizer.step()

        # Record results
        batch_losses["loss"].append(loss.item())
        batch_losses["loss_intent"].append(loss_intent.item())

        if itern % args.print_freq == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                f"loss_intent = {np.mean(batch_losses['loss_intent']): .4f}"
            )
        intent_prob = torch.sigmoid(intent_logit)
        recorder.train_intent_batch_update(
            itern,
            data,
            gt_intent.detach().cpu().numpy(),
            gt_intent_prob.detach().cpu().numpy(),
            intent_prob.detach().cpu().numpy(),
            loss.item(),
            loss_intent.item(),
        )

    epoch_loss["loss_intent"].append(np.mean(batch_losses["loss_intent"]))

    recorder.train_intent_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f"LearningRate", optimizer.param_groups[-1]["lr"], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f"Losses/{key}", np.mean(val), epoch)

    return epoch_loss


def train_traj(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
) -> None:
    """Trains the model for the trajectory task.

    :param model: The model to train.
    :type model: nn.Module
    :param optimizer: Pytorch optimizer to use.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Learning rate scheduler.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler
    :param train_loader: Train set dataloader.
    :type train_loader: DataLoader
    :param val_loader: Validation set dataloader.
    :type val_loader: DataLoader
    :param args: Training arguments.
    :type args: DefaultArguments
    :param recorder: Recorder object to log metrics.
    :type recorder: RecordResults
    :param writer: Tensorboard logging object.
    :type writer: SummaryWriter
    """
    # Handles some form of resumption of training.
    resume_info = ResumeTrainingInfo.load_resume_info(args.checkpoint_path)
    if resume_info is None:
        resume_info = ResumeTrainingInfo(args.epochs + 1, 1)
        resume_info.dump_resume_info(args.checkpoint_path)
    else:
        model, optimizer, scheduler = resume_info.load_state_dicts(
            args.checkpoint_path, model, optimizer, scheduler
        )
    pos_weight = torch.tensor(args.intent_positive_weight).to(
        DEVICE
    )  # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    # NOTE: Chris: I suspect that the NaN values may be due to the implementation of the loss functions.
    criterions = {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight
        ).to(DEVICE),
        "MSELoss": torch.nn.MSELoss(reduction="none").to(DEVICE),
        "BCELoss": torch.nn.BCELoss().to(DEVICE),
        "CELoss": torch.nn.CrossEntropyLoss().to(DEVICE),
        "L1Loss": torch.nn.L1Loss().to(DEVICE),
    }
    epoch_loss = {"loss_intent": [], "loss_traj": []}

    for epoch in range(resume_info.current_epoch, args.epochs + 1):
        niters = np.ceil(len(train_loader.dataset) / args.batch_size)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_traj_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            criterions,
            epoch_loss,
            train_loader,
            args,
            recorder,
            writer,
        )
        if not isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # Purge cache.
        gc.collect()
        torch.cuda.empty_cache()

        if epoch % 1 == 0:
            print(
                f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}, "
                f"loss_traj = {np.mean(epoch_loss['loss_traj']): .4f}"
            )

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = np.ceil(len(val_loader.dataset) / args.batch_size)
            recorder.eval_epoch_reset(epoch, niters)
            validate_traj(epoch, model, val_loader, args, recorder, writer)

        # Save training resumption info.
        torch.save(model.state_dict(), f"{args.checkpoint_path}/latest.pth")
        torch.save(optimizer.state_dict(), f"{args.checkpoint_path}/latest_optim.pth")
        torch.save(scheduler.state_dict(), f"{args.checkpoint_path}/latest_sched.pth")
        resume_info.current_epoch += 1
        resume_info.dump_resume_info(args.checkpoint_path)


def train_traj_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterions: dict[str, nn.Module],
    epoch_loss: dict[str, np.floating | float],
    dataloader: DataLoader,
    args: DefaultArguments,
    recorder: RecordResults,
    writer: SummaryWriter,
) -> dict[str, np.floating | float]:
    """Trains the model over 1 epoch.

    :param epoch: Current epoch value.
    :type epoch: int
    :param model: Model to train for 1 epoch.
    :type model: nn.Module
    :param optimizer: PyTorch optimizer used for training.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Learning rate scheduler used for training.
    :type scheduler: torch.optim.lr_scheduler.LRScheduler
    :param criterions: List of criterion names and callable criterions used to calculate loss.
    :type criterions: dict[str, nn.Module]
    :param epoch_loss: Dictionary containing various loss types. This object is used as a return
        value for the function.
    :type epoch_loss: dict
    :param args: Training arguments.
    :type args: DefaultArguments
    :param recorder: Recorder object to log metrics.
    :type recorder: RecordResults
    :param writer: Tensorboard logging object.
    :type writer: SummaryWriter

    :returns: Losses for the epoch.
    :rtype: dict[str, np.floating | float]
    """
    model.train()
    batch_losses = collections.defaultdict(list)

    niters: int = int(np.ceil(len(dataloader.dataset) / args.batch_size))
    for itern, data in enumerate(dataloader):
        # with torch.autograd.set_detect_anomaly(True):
        if CUDA and scaler:
            # Automatic mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                optimizer.zero_grad()
                traj_pred = model(data)
                traj_gt: torch.Tensor = (
                    data["bboxes"][:, args.observe_length :, :]
                    .type(FloatTensor)
                    .to(DEVICE)
                )
                loss_traj = torch.tensor(0.0).type(torch.bfloat16).to(DEVICE)
                if "bbox_l1" in args.traj_loss:
                    loss_bbox_l1 = criterions["L1Loss"](traj_pred, traj_gt)
                    batch_losses["loss_bbox_l1"].append(loss_bbox_l1.item())
                    loss_traj += loss_bbox_l1
                if "bbox_l2" in args.traj_loss:
                    loss_bbox_l2 = torch.mean(criterions["MSELoss"](traj_pred, traj_gt))
                    batch_losses["loss_bbox_l2"].append(loss_bbox_l2.item())
                    loss_traj += loss_bbox_l2

                loss = args.loss_weights["loss_traj"] * loss_traj
                # loss_traj = torch.mean(criterions["L1Loss"](traj_pred, traj_gt))
                # batch_losses["loss_bbox_l1"].append(loss_traj.item())
                # loss = loss_traj * args.loss_weights["loss_traj"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            scaler.update()

        else:
            optimizer.zero_grad()
            traj_pred = model(data)
            # intent_pred: sigmoid output, (0, 1), bs
            # traj_pred: logit, bs x ts x 4

            traj_gt = data["bboxes"][:, args.observe_length :, :].type(FloatTensor)
            bs, ts, _ = traj_gt.shape
            # center: bs x ts x 2
            # traj_center_gt = torch.cat((((traj_gt[:, :, 0] + traj_gt[:, :, 2]) / 2).unsqueeze(-1),
            #                             ((traj_gt[:, :, 1] + traj_gt[:, :, 3]) / 2).unsqueeze(-1)), dim=-1)
            # traj_center_pred = torch.cat((((traj_pred[:, :, 0] + traj_pred[:, :, 2]) / 2).unsqueeze(-1),
            #                               ((traj_pred[:, :, 1] + traj_pred[:, :, 3]) / 2).unsqueeze(-1)), dim=-1)

            loss_traj = torch.tensor(0.0).type(FloatTensor)
            if "bbox_l1" in args.traj_loss:
                loss_bbox_l1 = torch.mean(criterions["L1Loss"](traj_pred, traj_gt))
                batch_losses["loss_bbox_l1"].append(loss_bbox_l1.item())
                loss_traj += loss_bbox_l1
            if "bbox_l2" in args.traj_loss:
                loss_bbox_l2 = torch.mean(criterions["MSELoss"](traj_pred, traj_gt))
                batch_losses["loss_bbox_l2"].append(loss_bbox_l2.item())
                loss_traj += loss_bbox_l2

            loss = args.loss_weights["loss_traj"] * loss_traj
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()

        # Record results
        batch_losses["loss"].append(loss.item())
        batch_losses["loss_traj"].append(loss_traj.item())

        if itern % args.print_freq == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                f"loss_traj = {np.mean(batch_losses['loss_traj']): .4f}, "
            )
        recorder.train_traj_batch_update(
            itern,
            data,
            traj_gt.type(FloatTensor).detach().cpu().numpy(),
            traj_pred.type(FloatTensor).detach().cpu().numpy(),
            loss.item(),
            loss_traj.item(),
        )

    epoch_loss["loss_traj"].append(np.mean(batch_losses["loss_traj"]))

    recorder.train_traj_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f"LearningRate", optimizer.param_groups[-1]["lr"], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f"Losses/{key}", np.mean(val), epoch)

    return epoch_loss


def train_driving(
    model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer
):
    pos_weight = torch.tensor(args.intent_positive_weight).to(
        DEVICE
    )  # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight
        ).to(DEVICE),
        "MSELoss": torch.nn.MSELoss(reduction="none").to(DEVICE),
        "BCELoss": torch.nn.BCELoss().to(DEVICE),
        "CELoss": torch.nn.CrossEntropyLoss().to(DEVICE),
        "L1Loss": torch.nn.L1Loss(reduction="mean").to(DEVICE),
    }
    epoch_loss = {"loss_driving": [], "loss_driving_speed": [], "loss_driving_dir": []}

    for epoch in range(1, args.epochs + 1):
        niters = np.ceil(len(train_loader.dataset) / args.batch_size)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_driving_epoch(
            epoch,
            model,
            optimizer,
            criterions,
            epoch_loss,
            train_loader,
            args,
            recorder,
            writer,
        )
        scheduler.step()

        if epoch % 1 == 0:
            print(
                f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                f"loss_driving_speed = {np.mean(epoch_loss['loss_driving_speed']): .4f}, "
                f"loss_driving_dir = {np.mean(epoch_loss['loss_driving_dir']): .4f}"
            )

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = np.ceil(len(val_loader.dataset) / args.batch_size)
            recorder.eval_epoch_reset(epoch, niters)
            validate_driving(epoch, model, val_loader, args, recorder, writer)

        torch.save(model.state_dict(), args.checkpoint_path + f"/latest.pth")


def train_driving_epoch(
    epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer
):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = np.ceil(len(dataloader.dataset) / args.batch_size)
    for itern, data in enumerate(dataloader):
        optimizer.zero_grad()
        pred_speed_logit, pred_dir_logit = model(data)
        lbl_speed = data["label_speed"].type(LongTensor)  # bs x 1
        lbl_dir = data["label_direction"].type(LongTensor)  # bs x 1
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

        loss_driving = torch.tensor(0.0).type(FloatTensor)
        if "cross_entropy" in args.driving_loss:
            loss_driving_speed = torch.mean(
                criterions["CELoss"](pred_speed_logit, lbl_speed)
            )
            loss_driving_dir = torch.mean(criterions["CELoss"](pred_dir_logit, lbl_dir))
            # loss_bbox_l1 = torch.mean(criterions['L1Loss'](traj_pred, traj_gt))
            batch_losses["loss_driving_speed"].append(loss_driving_speed.item())
            batch_losses["loss_driving_dir"].append(loss_driving_dir.item())
            loss_driving += loss_driving_speed
            loss_driving += loss_driving_dir

        loss = args.loss_weights["loss_driving"] * loss_driving
        loss.backward()
        optimizer.step()

        # Record results
        batch_losses["loss"].append(loss.item())
        batch_losses["loss_driving"].append(loss_driving.item())

        if itern % args.print_freq == 0:
            print(
                f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                f"loss_driving_speed = {np.mean(batch_losses['loss_driving_speed']): .4f}, "
                f"loss_driving_dir = {np.mean(batch_losses['loss_driving_dir']): .4f}"
            )
        recorder.train_driving_batch_update(
            itern,
            data,
            lbl_speed.detach().cpu().numpy(),
            lbl_dir.detach().cpu().numpy(),
            pred_speed_logit.detach().cpu().numpy(),
            pred_dir_logit.detach().cpu().numpy(),
            loss.item(),
            loss_driving_speed.item(),
            loss_driving_dir.item(),
        )

        # if itern >= 10:
        #     break

    epoch_loss["loss_driving"].append(np.mean(batch_losses["loss_driving"]))
    epoch_loss["loss_driving_speed"].append(np.mean(batch_losses["loss_driving_speed"]))
    epoch_loss["loss_driving_dir"].append(np.mean(batch_losses["loss_driving_dir"]))

    recorder.train_driving_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f"LearningRate", optimizer.param_groups[-1]["lr"], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f"Losses/{key}", np.mean(val), epoch)

    return epoch_loss
