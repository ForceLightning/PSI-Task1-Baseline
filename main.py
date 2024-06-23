from datetime import datetime
import json
import os
from typing import Any

import numpy as np
from sklearn.model_selection import ParameterSampler
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from opts import get_opts
from test import get_test_traj_gt, predict_driving, predict_intent, predict_traj
from train import train_driving, train_intent, train_traj
from utils.args import DefaultArguments
from utils.evaluate_results import evaluate_driving, evaluate_intent, evaluate_traj
from utils.get_test_intent_gt import get_intent_gt, get_test_driving_gt
from utils.log import RecordResults


def main(args: DefaultArguments) -> tuple[float | np.floating[Any], float]:
    writer = SummaryWriter(args.checkpoint_path, comment=args.comment)
    recorder = RecordResults(args)
    """ 1. Load database """
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)
    args.steps_per_epoch = int(np.ceil(len(train_loader.dataset) / args.batch_size))

    """ 2. Create models """
    model, optimizer, scheduler = build_model(args)
    model = nn.DataParallel(model)
    metrics = {}

    # ''' 3. Train '''
    val_score, test_acc = 0.0, 0.0
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
            )
            val_gt_file = "./test_gt/val_intent_gt.json"
            if not os.path.exists(val_gt_file):
                get_intent_gt(val_loader, val_gt_file, args)
            predict_intent(model, val_loader, args, dset="val")
            val_score, val_f1, val_mAcc = evaluate_intent(
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
                test_acc, test_f1, test_mAcc = evaluate_intent(
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
            )
            val_gt_file = "./test_gt/val_traj_gt.json"
            if not os.path.exists(val_gt_file):
                get_test_traj_gt(model, val_loader, args, dset="val")
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
            )

            val_gt_file = "./test_gt/val_driving_gt.json"
            if not os.path.exists(val_gt_file):
                get_test_driving_gt(model, val_loader, args, dset="val")
            predict_driving(model, val_loader, args, dset="val")
            val_score = evaluate_driving(
                val_gt_file,
                args.checkpoint_path + "/results/val_driving_pred.json",
                args,
            )
            metrics = {"hparam/val_score": val_score}

    hparams = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "kernel_size": args.kernel_size,
        "n_layers": args.n_layers,
    }

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    return val_score, test_acc


if __name__ == "__main__":
    args = get_opts()
    # Task
    args.task_name = args.task_name if args.task_name else "ped_traj"
    args.persist_dataloader = True

    match args.task_name:
        case "ped_intent":
            args.database_file = "intent_database_train.pkl"
            args.intent_model = True
            args.batch_size = [256]
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
            args.traj_loss = ["bbox_l2"]
            # args.traj_loss = ["bbox_huber"]
            # args.batch_size = [256]
            # args.batch_size = [64]
            args.batch_size = [256]
            args.predict_length = 45
            args.model_name = "tcn_traj_bbox"
            # args.model_name = "tcn_traj_bbox_int"
            # args.model_name = "tcn_traj_global"
            # args.model_name = "tcan_traj_bbox"
            # args.model_name = "tcan_traj_bbox_int"
            # args.model_name = "tcan_traj_global"
            args.loss_weights = {
                "loss_intent": 0.0,
                "loss_traj": 1.0,
                "loss_driving": 0.0,
            }
            args.load_image = False
            # args.backbone = ""
            # args.freeze_backbone = False
            # args.load_image = True
            args.backbone = "resnet50"
            args.freeze_backbone = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE

        case "driving_decision":
            args.database_file = "driving_database_train.pkl"
            args.driving_loss = ["cross_entropy"]
            # args.batch_size = [64]
            args.batch_size = [256]
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
            # args.seq_overlap_rate = 0.1  # overlap rate for train/val set
            args.seq_overlap_rate = 1
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
            args.predict_length = 1

    args.observe_length = 15
    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = "enlarge"
    args.normalize_bbox = None
    # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
    # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]

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
        # "lr": [3e-3],
        "lr": [3e-3],
        "batch_size": args.batch_size,
        "epochs": [50],
        "n_layers-kernel_size": [(4, 2)],
    }

    n_random_samples = 60

    parameter_samples = list(
        ParameterSampler(hyperparameter_list, n_iter=n_random_samples)
    )

    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    best_val_accuracy = 0.0
    best_hyperparameters = None

    checkpoint_path = args.checkpoint_path

    for params in parameter_samples:
        args.lr = params["lr"]
        args.batch_size = params["batch_size"]
        args.epochs = params["epochs"]
        args.n_layers, args.kernel_size = params["n_layers-kernel_size"]

        # Record
        now = datetime.now()
        if args.comment is None:
            time_folder = now.strftime("%Y%m%d%H%M%S")
        else:
            time_folder = args.comment
        if args.checkpoint_path == "./ckpts" and args.resume == "":
            args.checkpoint_path = os.path.join(
                checkpoint_path,
                args.task_name,
                args.dataset,
                args.model_name,
                time_folder,
            )
        elif args.checkpoint_path == "./ckpts":
            args.checkpoint_path = os.path.dirname(args.resume)
        elif args.checkpoint_path != "./ckpts" and args.resume != "":
            assert os.path.abspath(args.checkpoint_path) == os.path.abspath(
                os.path.dirname(args.resume)
            ), f"checkpoint path and resume path directories do not match, {os.path.abspath(args.checkpoint_path)}, {os.path.abspath(os.path.dirname(args.resume))}"

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
        val_accuracy, test_accuracy = main(args)
        print("Validation Accuracy:", val_accuracy)
        print("Test Accuracy:", test_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparameters = params

    print("Best Validation Accuracy:", best_val_accuracy)
    print("Best Hyperparameters:", best_hyperparameters)
