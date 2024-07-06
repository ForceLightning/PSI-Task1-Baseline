import json
import os
import warnings
from argparse import ArgumentParser, Namespace
from math import ceil
from typing import Any

import numpy as np
import torch
from numpy import typing as npt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from typing_extensions import Literal

from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from models.model_interfaces import ITSTransformerWrapper
from models.traj_modules.model_transformer_traj_bbox import TransformerTrajBbox
from utils import get_test_intent_gt
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
) -> RecordResults:
    """Validate the intention model.

    :param int epoch: Current epoch.
    :param nn.Module model: Model to validate.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param RecordResults recorder: Recorder for the results.
    :param SummaryWriter writer: Tensorboard writer object.

    :return: Recorder for the results.
    :rtype: RecordResults
    """
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
    """Test the intention model.

    :param int epoch: Current epoch.
    :param nn.Module model: Model to test.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param RecordResults recorder: Recorder for the results.
    :param SummaryWriter writer: Tensorboard writer object.

    :return: Recorder for the results.
    :rtype: RecordResults
    """
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
    dset: Literal["train", "val", "test"] = "test",
) -> None:
    """Predict and save prediction of intention for each sample in the dataset.

    :param nn.Module model: Model to predict intention.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param dset: Dataset to predict intention.
    :type dset: Literal["train", "val", "test"]

    :return: None
    """
    _ = model.eval()
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
        os.path.join(args.checkpoint_path, "results", f"{dset}_intent_pred.json"), "w"
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
    """Validate the trajectory model.

    :param int epoch: Current epoch.
    :param nn.Module model: Model to validate.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param RecordResults recorder: Recorder for the results.
    :param SummaryWriter writer: Tensorboard writer object.
    :param nn.Module criterion: Loss function for the model.

    :return: Validation loss.
    :rtype: float
    """
    _ = model.eval()
    niters: int = len(dataloader)
    num_samples: int = len(dataloader.dataset)
    val_losses: list[float] = []
    for itern, data in enumerate(dataloader):
        traj_pred: torch.Tensor
        traj_gt: torch.Tensor = data["bboxes"][:, args.observe_length :, :].type(
            FloatTensor
        )
        val_loss: float

        # Extract the inner model from the DataParallel module so that we may run
        # the generate() method on a Transformer.
        inner_model = getattr(model, "module", model)
        if isinstance(inner_model, TransformerTrajBbox):
            # WARNING: Without setting the num_parallel samples to 1, memory usage
            # explodes.
            old_num_parallel_samples = inner_model.config.num_parallel_samples
            inner_model.config.num_parallel_samples = 1
            traj_pred = inner_model.generate(data)
            val_loss = criterion(traj_pred, traj_gt).item()
            inner_model.config.num_parallel_samples = old_num_parallel_samples
        else:
            traj_pred = model(data)
            val_loss = (
                criterion(traj_pred, traj_gt).item()
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
    dset: Literal["train", "val", "test"] = "test",
) -> None:
    """Predict and save prediction of trajectory for each sample in the dataset.

    :param nn.Module model: Model to predict trajectory.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param dset: Dataset to predict trajectory.
    :type dset: Literal["train", "val", "test"]

    :return: None
    """
    _ = model.eval()
    dt = {}
    num_batches: int = ceil(len(dataloader.dataset) / dataloader.batch_size)
    for itern, data in tqdm(
        enumerate(dataloader), total=num_batches, desc="Test batches"
    ):
        # TODO: (chris) Allow for evaluation of transformer outputs, see
        # `test.py:validate_traj` for info. Otherwise the following line will just
        # return a tuple and we have no error handling :)
        inner_model = getattr(model, "module", model)
        traj_pred: torch.Tensor
        if isinstance(inner_model, ITSTransformerWrapper):
            old_num_parallel_samples = inner_model.config.num_parallel_samples
            inner_model.config.num_parallel_samples = 1
            traj_pred = inner_model.generate(data)
            inner_model.config.num_parallel_samples = old_num_parallel_samples
        else:
            traj_pred = model(data)

        for i in range(len(data["frames"])):  # for each sample in a batch
            vid: str = data["video_id"][i]  # str list, bs x 60
            pid: str = data["ped_id"][i]  # str list, bs x 60

            # int list, bs x 15, observe 0~14, predict 15th intent
            fid: int = (data["frames"][i][-1] + 1).item()

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
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    dset: str = "test",
) -> None:
    """Get ground truth trajectory for each sample in the dataset.

    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param str dset: Dataset to predict driving decision.

    :return: None
    """
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
    with open(
        os.path.join(args.dataset_root_path, "test_gt", f"{dset}_traj_gt.json"), "w"
    ) as f:
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
) -> float:
    """Validate the driving decision model.

    :param int epoch: Current epoch.
    :param nn.Module model: Model to validate.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param RecordResults recorder: Recorder for the results.
    :param SummaryWriter writer: Tensorboard writer object.
    :param nn.Module criterion: Loss function for the model.

    :return: Validation loss.
    :rtype: float
    """
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
def predict_driving(
    model: nn.Module,
    dataloader: DataLoader[Any],
    args: DefaultArguments,
    dset: Literal["train", "val", "test"] = "test",
) -> None:
    """Predict and save prediction of driving decision for each sample in the dataset.

    :param nn.Module model: Model to predict driving decision.
    :param DataLoader[Any] dataloader: DataLoader for the dataset.
    :param DefaultArguments args: Arguments for the model.
    :param dset: Dataset to predict driving decision.
    :type dset: Literal["train", "val", "test"]

    :return: None
    """
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


def load_args(dataset_root_path: str, ckpt_path: str) -> DefaultArguments:
    ckpt_path = os.path.normpath(ckpt_path)

    def _load_args(args: DefaultArguments, args_path: str) -> DefaultArguments:
        if os.path.exists(args_path):
            with open(args_path, "r", encoding="utf-8") as f:
                args_dict: dict[str, Any] = json.load(f)
                attr_bar = tqdm(args_dict.items(), desc="Attributes", leave=False)
                for k, v in attr_bar:
                    if hasattr(args, k):
                        attr_bar.set_description(f"Attribute: {k}")
                        if k in ["video_splits"]:
                            path: str = os.path.normpath(v)
                            if path.split(os.sep)[0] != os.path.normpath(
                                dataset_root_path
                            ):
                                setattr(args, k, os.path.join(dataset_root_path, path))
                            else:
                                setattr(args, k, path)
                        else:
                            setattr(args, k, v)
                    else:
                        warnings.warn(f"Attribute {k} not found in DefaultArguments.")
        else:
            raise FileNotFoundError(f"File not found: {args_path}")
        return args

    args: DefaultArguments = DefaultArguments()

    for args_fn in ["args.json", "args.txt", "args"]:
        args_path: str = os.path.join(ckpt_path, args_fn)
        try:
            args = _load_args(args, args_path)
            return args
        except FileNotFoundError as e:
            continue

    args.checkpoint_path = ckpt_path

    warnings.warn("Using default arguments.")
    return args


def load_model_state_dict(ckpt_path: str, model: nn.Module) -> nn.Module:
    if os.path.exists(model_ckpt := os.path.join(ckpt_path, "latest.pth")):
        incomp_keys = model.load_state_dict(torch.load(model_ckpt))
        if incomp_keys.unexpected_keys or incomp_keys.missing_keys:
            warnings.warn(str(incomp_keys))

    return model


def predict(
    dataset_root_path: str, ckpt_path: str, test_loader: DataLoader[Any] | None = None
) -> tuple[DataLoader[Any], DefaultArguments]:
    args = load_args(dataset_root_path, ckpt_path)

    # NOTE: This is a stupid hack for the `lag_sequence` parameter in transformer
    # configs.
    if "transformer" in args.model_name:
        args.observe_length += 1
        args.max_track_size += 1

    # Load models
    model, _, _ = build_model(args)
    if args.compile_model:
        model = torch.compile(
            model, options={"triton.cudagraphs": True}, fullgraph=True
        )
    model = nn.DataParallel(model)

    model = load_model_state_dict(ckpt_path, model)

    # Load pickled DB
    # NOTE: The hack continues
    if test_loader is None or "transformer" in args.model_name:
        if not os.path.exists(os.path.join(args.database_path, args.database_file)):
            create_database(args)
        else:
            print("Database exists!")

        _, _, test_loader = get_dataloader(args, load_test=True)
        assert test_loader is not None, "Cannot perform tests without test dataloader!"

    # Create results dir
    if not os.path.exists(os.path.join(args.checkpoint_path, "results")):
        os.makedirs(os.path.join(args.checkpoint_path, "results"))

    match args.task_name:
        case "ped_intent":
            predict_intent(model, test_loader, args, dset="test")
        case "ped_traj":
            predict_traj(model, test_loader, args, dset="test")
        case "driving_decision":
            predict_driving(model, test_loader, args, "test")

    return test_loader, args


def main(args: Namespace):
    # NOTE: Now this one is a silly way to create a single test loader for all models.
    test_loader: DataLoader[Any] | None = None

    mbar = tqdm(args.model_checkpoints, desc="Model checkpoints", leave=False)
    for ckpt_path in mbar:
        if os.path.exists(full_ckpt_path := os.path.join(args.task_path, ckpt_path)):
            model_name = os.path.normpath(ckpt_path).split(os.sep)[-2]
            mbar.set_description(f"Model checkpoint: {model_name}")

            try:
                test_loader, former_args = predict(
                    args.dataset_root_path,
                    os.path.normpath(full_ckpt_path),
                    test_loader,
                )

                test_gt_file = ""
                if "ped_intent" in args.task_path:
                    test_gt_file = os.path.join(
                        args.dataset_root_path, "test_gt", "test_intent_gt.json"
                    )
                    if not os.path.exists(test_gt_file):
                        get_test_intent_gt.get_intent_gt(
                            test_loader, test_gt_file, former_args
                        )
                        mbar.write(f"Saved test intent ground truth to {test_gt_file}")
                    else:
                        mbar.write(f"Test intent ground truth exists at {test_gt_file}")
                elif "ped_traj" in args.task_path:
                    test_gt_file = os.path.join(
                        args.dataset_root_path, "test_gt", "test_traj_gt.json"
                    )
                    if not os.path.exists(test_gt_file):
                        get_test_traj_gt(test_loader, former_args, dset="test")
                        mbar.write(f"Saved test intent ground truth to {test_gt_file}")
                    else:
                        mbar.write(f"Test intent ground truth exists at {test_gt_file}")
                elif "driving_decision" in args.task_path:
                    test_gt_file = os.path.join(
                        args.dataset_root_path, "test_gt", "test_driving_gt.json"
                    )
                    if not os.path.exists(test_gt_file):
                        get_test_intent_gt.get_test_driving_gt(
                            test_loader, former_args, dset="test"
                        )
                        mbar.write(f"Saved test intent ground truth to {test_gt_file}")
                    else:
                        mbar.write(f"Test intent ground truth exists at {test_gt_file}")

            except Exception as e:
                warnings.warn(f"Model: {model_name}, Error: {e}")
                continue
        else:
            warnings.warn(f"Checkpoint path {ckpt_path} does not exist!")
            continue


if __name__ == "__main__":
    argparser = ArgumentParser()

    _ = argparser.add_argument(
        "-r",
        "--dataset_root_path",
        type=str,
        default="../",
        help="Path to the [r]oot of the dataset",
    )
    _ = argparser.add_argument(
        "-t",
        "--task_path",
        type=str,
        help="Path to the root [t]ask checkpoint directory",
        default="../ckpts/ped_intent/PSI2.0/",
        required=True,
    )
    _ = argparser.add_argument(
        "-m",
        "--model_checkpoints",
        type=str,
        nargs="+",
        help="[M]odel checkpoint paths from the root task checkpoint directory, e.g. if the full path is `../ckpts/ped_intent/PSI2.0/lstm_int_bbox/20240124135257/`, and the task path is `../ckpts/ped_intent/PSI2.0/`, then the model checkpoint paths are `lstm_int_bbox/20240124135257/`",
        required=True,
    )

    args = argparser.parse_args()
    main(args)
