"""Main script for training and testing the model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterSampler
from torch.utils.tensorboard.writer import SummaryWriter

from data.prepare_data import get_dataloader
from database.create_database import create_database
from eval import get_test_traj_gt, predict_driving, predict_intent, predict_traj
from models.build_model import build_model
from opts import get_opts
from train import train_driving, train_intent, train_traj
from utils.args import DefaultArguments
from utils.evaluate_results import evaluate_driving, evaluate_intent, evaluate_traj
from utils.get_test_intent_gt import get_intent_gt, get_test_driving_gt
from utils.log import RecordResults


def train(args: DefaultArguments) -> tuple[float | np.floating[Any], float]:
    """Main function for training and testing the model.

    :param DefaultArguments args: The training arguments.
    :return: The validation score and test accuracy.
    """
    writer = SummaryWriter(args.checkpoint_path, comment=args.comment)
    recorder = RecordResults(args)
    if "transformer" in args.model_name:  # handles "lag" in the sequence
        args.observe_length += 1
        args.max_track_size += 1
    """ 1. Load database """
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)
    args.steps_per_epoch = int(np.ceil(len(train_loader.dataset) / args.batch_size))

    """ 2. Create models """
    model, optimizer, scheduler = build_model(args)
    if args.compile_model:
        model = torch.compile(
            model, options={"triton.cudagraphs": True}, fullgraph=True
        )
    model = nn.DataParallel(model)
    metrics = {}

    # ''' 3. Train '''
    val_score, test_acc = 0.0, 0.0
    prof: torch.profiler.profile | None = None
    if args.profile_execution:
        if not os.path.exists(os.path.join(args.checkpoint_path, "log")):
            os.makedirs(os.path.join(args.checkpoint_path, "log"))

        def trace_handler(prof: torch.profiler.profile):
            prof.export_chrome_trace(
                os.path.join(
                    args.checkpoint_path,
                    "log",
                    f"profiling_results_{prof.step_num}.pt.trace.json",
                )
            )
            _ = torch.profiler.tensorboard_trace_handler(
                dir_name=os.path.join(args.checkpoint_path, "log")
            )

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # record_shapes=True,
            profile_memory=True,
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=2, repeat=1, skip_first=8
            ),
            on_trace_ready=trace_handler,
        )
        prof.start()

    match args.task_name:
        case "ped_intent":
            train_intent(
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                args,
                recorder,
                writer,
                prof,
            )
            val_gt_file = os.path.join(
                args.dataset_root_path, "test_gt/val_intent_gt.json"
            )
            if not os.path.exists(val_gt_file):
                get_intent_gt(val_loader, val_gt_file, args)
            predict_intent(model, val_loader, args, dset="val")
            val_score, val_f1, val_mAcc, _ = evaluate_intent(
                val_gt_file, args.checkpoint_path + "/results/val_intent_pred", args
            )
            metrics = {
                "hparam/val_accuracy": val_score,
                "hparam/val_f1": val_f1,
                "hparam/val_mAcc": val_mAcc,
            }

            # ''' 4. Test '''
            if test_loader is not None:
                test_gt_file = "./test_gt/test_intent_gt.json"
                if not os.path.exists(test_gt_file):
                    get_intent_gt(test_loader, test_gt_file, args)
                predict_intent(model, test_loader, args, dset="test")
                test_acc, test_f1, test_mAcc, _ = evaluate_intent(
                    test_gt_file,
                    args.checkpoint_path + "/results/test_intent_pred",
                    args,
                )
                metrics["hparam/test_accuracy"] = test_acc
                metrics["hparam/test_f1"] = test_f1
                metrics["hparam/test_mAcc"] = test_mAcc

        case "ped_traj":
            train_traj(
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                args,
                recorder,
                writer,
                prof,
            )
            val_gt_file = os.path.join(
                args.dataset_root_path, "test_gt/val_traj_gt.json"
            )
            if not os.path.exists(val_gt_file):
                get_test_traj_gt(val_loader, args, dset="val")
            predict_traj(model, val_loader, args, dset="val")
            val_score = evaluate_traj(
                val_gt_file,
                args.checkpoint_path + "/results/val_traj_pred.json",
                args,
            )
            metrics = {"hparam/val_score": val_score}

        case "driving_decision":
            train_driving(
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                args,
                recorder,
                writer,
                prof,
            )

            val_gt_file = os.path.join(
                args.dataset_root_path, "test_gt/val_driving_gt.json"
            )
            if not os.path.exists(val_gt_file):
                get_test_driving_gt(val_loader, val_gt_file, args)
            predict_driving(model, val_loader, args, dset="val")
            val_score = evaluate_driving(
                val_gt_file,
                args.checkpoint_path + "/results/val_driving_pred.json",
                args,
            )
            metrics = {"hparam/val_score": val_score}

    if args.profile_execution and prof is not None:
        prof.export_chrome_trace(
            os.path.join(args.checkpoint_path, "log", "profiling_results.json")
        )
        prof.stop()

    hparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "kernel_size": args.kernel_size,
        "n_layers": args.n_layers,
    }

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    return val_score, test_acc


def main(args: DefaultArguments):
    args.task_name = args.task_name if args.task_name else "ped_traj"
    args.persist_dataloader = True

    match args.task_name:
        case "ped_intent":
            args.database_file = "intent_database_train.pkl"
            args.intent_model = True
            args.batch_size = [256]  # type: ignore[reportAttributeAccessIssue]
            # intent prediction
            args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
            args.intent_type = "mean"  # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
            args.intent_loss = ["bce"]  # type: ignore[reportAttributeAccessIssue]
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
            # args.traj_loss = ["nll"]
            # args.traj_loss = ["bbox_huber"]
            args.traj_loss = ["bbox_l2"]
            # args.traj_loss = ["bbox_l1"]
            # args.batch_size = [256]
            # args.batch_size = [64]
            args.batch_size = [256]  # type: ignore[reportAttributeAccessIssue]
            args.predict_length = 45
            # args.model_name = "lstm_traj_bbox"
            # args.model_name = "tcn_traj_bbox"
            # args.model_name = "tcn_traj_bbox_int"
            # args.model_name = "tcn_traj_bbox_pose"
            # args.model_name = "tcn_traj_global"
            # args.model_name = "tcan_traj_bbox"
            # args.model_name = "tcan_traj_bbox_int"
            args.model_name = "tcan_traj_bbox_pose"
            # args.model_name = "tcan_traj_global"
            # args.model_name = "transformer_traj_bbox"
            # args.model_name = "transformer_traj_bbox_pose"
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
            args.normalize_bbox = "divide_image_size"

        case "driving_decision":
            args.database_file = "driving_database_train.pkl"
            args.driving_loss = ["cross_entropy"]
            # args.batch_size = [64]
            args.batch_size = [256]  # type: ignore[reportAttributeAccessIssue]
            # args.model_name = "reslstm_driving_global"
            args.model_name = "restcn_driving_global"
            args.loss_weights = {
                "loss_intent": 0.0,
                "loss_traj": 0.0,
                "loss_driving": 1.0,
            }
            args.load_image = False
            args.backbone = "resnet50"
            args.freeze_backbone = True
            args.seq_overlap_rate = 1
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
            args.predict_length = 1

    args.observe_length = 15
    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = "enlarge"

    # if args.load_image:
    # args.backbone = "resnet"
    # args.freeze_backbone = False
    # else:
    # args.backbone = None
    # args.freeze_backbone = False

    # Train
    # hyperparameter_list = {
    #     "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    #     "batch_size": [64, 128, 256, 512, 1024],
    #     "epochs": [50],
    #     "n_layers-kernel_size": [(2, 8), (3, 3), (4, 2)],
    # }

    hyperparameter_list = {
        # "lr": [3e-2],
        "lr": [3e-3],
        # "lr": [1e-1],
        # "lr": [1e-2],
        # "lr": [1e-5],
        "batch_size": args.batch_size,
        "epochs": [50],
        "n_layers-kernel_size": [(4, 2)],
    }

    n_random_samples = 60

    parameter_samples: list[
        dict[
            {
                "lr": float,
                "batch_size": int,
                "epochs": int,
                "n_layers-kernel_size": tuple[int, int],
            }
        ]
    ] = list(ParameterSampler(hyperparameter_list, n_iter=n_random_samples))

    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    best_val_accuracy = 0.0
    best_hyperparameters = None

    checkpoint_path: str = os.path.join(args.dataset_root_path, args.checkpoint_path)
    args.checkpoint_path = checkpoint_path

    for params in parameter_samples:
        args.lr = params["lr"]
        args.batch_size = params["batch_size"]
        args.epochs = params["epochs"]
        args.n_layers, args.kernel_size = params["n_layers-kernel_size"]

        # Record
        now = datetime.now()
        if args.comment == "":
            ckpt_directory = now.strftime("%Y%m%d%H%M%S")
        else:
            ckpt_directory = args.comment

        args.checkpoint_path = args.checkpoint_path.rstrip("/")
        if os.path.split(args.checkpoint_path)[-1] == "ckpts":
            if args.resume == "":
                args.checkpoint_path = os.path.join(
                    checkpoint_path,
                    args.task_name,
                    args.dataset,
                    args.model_name,
                    ckpt_directory,
                )
            else:
                args.checkpoint_path = os.path.dirname(args.resume)
        else:
            full_ckpt_dir = os.path.abspath(args.checkpoint_path)
            full_resume_parent_dir = os.path.abspath(os.path.dirname(args.resume))
            assert (
                full_ckpt_dir == full_resume_parent_dir
            ), f"checkpoint path and resume path's directory does not match, {full_ckpt_dir}, {full_resume_parent_dir}"

        print(f"Checkpoint path: {args.checkpoint_path}")
        # TODO: Move args dumping to after model_opts are created.
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
            with open(
                os.path.join(args.checkpoint_path, "args.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(args.__dict__, f, indent=4)
        else:
            # Load arguments if resume is provided.
            with open(
                os.path.join(args.checkpoint_path, "args.json"), "r", encoding="utf-8"
            ) as f:
                args.__dict__ = json.load(f)

        result_path = os.path.join(args.checkpoint_path, "results")
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        print("Running with Parameters:", params)  # Print the current parameters
        val_accuracy, test_accuracy = train(args)
        print("Validation Accuracy:", val_accuracy)
        print("Test Accuracy:", test_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparameters = params

    print("Best Validation Accuracy:", best_val_accuracy)
    print("Best Hyperparameters:", best_hyperparameters)


if __name__ == "__main__":
    args = get_opts()
    main(args)
