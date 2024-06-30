"""Main script for training and testing the model.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from boxmot import BoTSORT, BYTETracker, DeepOCSORT
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import ParameterSampler
from torch.utils.tensorboard.writer import SummaryWriter
from ultralytics import YOLO
from yolo_tracking.boxmot.utils import ROOT
from yolo_tracking.tracking.track import run

from data.custom_dataset import YoloDataset
from data.prepare_data import (
    get_dataloader,
    get_video_dimensions,
    save_data_to_txt,
    visualise_annotations,
    visualise_intent,
)
from database.create_database import create_database
from eval import get_test_traj_gt, predict_driving, predict_intent, predict_traj
from models.build_model import build_model
from opts import get_opts
from train import train_driving, train_intent, train_traj
from utils.args import DefaultArguments
from utils.evaluate_results import evaluate_driving, evaluate_intent, evaluate_traj
from utils.get_test_intent_gt import get_intent_gt, get_test_driving_gt
from utils.log import RecordResults
from utils.plotting import (
    PosePlotter,
    draw_landmarks_on_image,
    overlay,
    road_lane_detection,
)


def main(args: DefaultArguments) -> tuple[float | np.float_, float]:
    """Main function for training and testing the model.

    :param DefaultArguments args: The training arguments.
    :return: The validation score and test accuracy.
    """
    # Set args.classes to 0 for pedestrian tracking
    args.classes = 0

    # If video source source is from test
    args.source = os.path.join(os.getcwd(), "PSI2.0_Test", "videos", "video_0147.mp4")

    file_name = args.source.split("\\")[-1].split(".")[0]

    model = YOLO("yolov8s.pt")
    tracker = DeepOCSORT(
        model_weights=Path("osnet_x0_25_msmt17.pt"),  # which ReID model to use
        device="cuda:0",
        fp16=False,
    )
    # If video source is from val
    args.source = os.path.join(os.getcwd(), "PSI_Videos", "videos", "video_0131.mp4")
    file_name = args.source.split("\\")[-1].split(".")[0]
    width, height = get_video_dimensions(args.source)
    run(args)

    bbox_holder, frames_holder, video_id = consolidate_yolo_data(width, height)
    save_data_to_txt(bbox_holder, frames_holder, video_id)
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

    example_data = YoloDataset(os.path.join(ROOT, "yolo_results_data"))

    # num_workers > 0 gives me error
    example_loader = torch.utils.data.DataLoader(
        example_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=None,
        num_workers=0,
    )

    yolo = YOLO("yolov8s-pose.pt")
    tracker = BoTSORT(
        model_weights=Path("osnet_x0_25_msmt17.pt"), device="cuda:0", fp16=True
    )

    pose_plotter = PosePlotter()

    vid = cv2.VideoCapture(args.source)

    # frame_number = 0
    #
    # while True:
    #     ret, im = vid.read()
    #
    #     if not ret :
    #         break
    #
    #     # Increment frame number
    #     frame_number += 1
    #     result = model.predict(im, verbose=False, classes=[0], conf=0.15)
    #     dets = []
    #     kp_dets = []
    #     result = result[0]
    #     # im = result.plot()
    #
    #     for k in result.keypoints:
    #         conf = k.conf
    #         kp = k.data # x, y, visibility - xy non-normalized
    #         kp_dets.append(kp.tolist())
    #
    #     for box in result.boxes:
    #         cls = int(box.cls[0].item())
    #         cords = box.xyxy[0].tolist()
    #         conf = box.conf[0].item()
    #         id = box.id
    #         dets.append([cords[0], cords[1], cords[2], cords[3], conf, cls])
    #
    #     dets = np.array(dets)
    #     kp_dets = np.array(kp_dets)
    #
    #     if len(dets) == 0:
    #         dets = np.empty((0, 6))
    #
    #     tracks = tracker.update(dets, im)
    #     if tracks.shape[0] != 0:
    #     #     x1 = tracks[0][0]
    #     #     y1 = tracks[0][1]
    #     #     x2 = tracks[0][2]
    #     #     y2 = tracks[0][3]
    #     #     id = tracks[0][4]
    #     #     conf = tracks[0][5]
    #     #     cls = tracks[0][6]
    #     #     with open(file_name + ".txt", 'a') as f:
    #     #         f.write(f"{int(id)} {x1} {y1} {x2} {y2} {conf} {int(cls)} {frame_number}\n")
    #
    #         inds = tracks[:, 7].astype('int') # float64 to int
    #         keypoints = kp_dets[inds]
    #
    #         pose_plotter.plot_keypoints(image=im, keypoints=keypoints)
    #         tracker.plot_results(im, show_trajectories=False)
    #
    #     cv2.imshow('BoxMOT detection', im)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord(' ') or key == ord('q'):
    #         break

    vid.release()
    cv2.destroyAllWindows()
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
            torch.profiler.tensorboard_trace_handler(
                os.path.join(args.checkpoint_path, "log")
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
                predict_intent(model, example_loader, args, dset="test")
                # Visualise specific bbox from specific frame fed into TCN for sanity check
                visualise_annotations(
                    os.path.join(ROOT, "yolo_results_data", "1.txt"), 0
                )
                visualise_intent(
                    os.path.join(ROOT, "runs", "track", "exp", "labels"),
                    os.path.join(os.getcwd(), "test_gt", "test_intent_pred"),
                    width,
                    height,
                )
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
                prof,
            )

            val_gt_file = os.path.join(
                args.dataset_root_path, "test_gt/val_driving_gt.json"
            )
            if not os.path.exists(val_gt_file):
                get_test_driving_gt(model, val_loader, args, dset="val")
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
            args.traj_loss = ["nll"]
            # args.traj_loss = ["bbox_huber"]
            # args.batch_size = [256]
            # args.batch_size = [64]
            args.batch_size = [256]
            args.predict_length = 45
            # args.model_name = "tcn_traj_bbox"
            # args.model_name = "tcn_traj_bbox_int"
            # args.model_name = "tcn_traj_global"
            # args.model_name = "tcan_traj_bbox"
            # args.model_name = "tcan_traj_bbox_int"
            # args.model_name = "tcan_traj_global"
            # args.model_name = "transformer_traj_bbox"
            args.model_name = "transformer_traj_bbox_pose"
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
        "epochs": [4],
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

    checkpoint_path: str = os.path.join(args.dataset_root_path, args.checkpoint_path)
    args.checkpoint_path = checkpoint_path

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
        if (
            args.checkpoint_path == os.path.join(args.dataset_root_path, "ckpts")
            or args.checkpoint_path == "./ckpts"
        ) and args.resume == "":
            args.checkpoint_path = os.path.join(
                checkpoint_path,
                args.task_name,
                args.dataset,
                args.model_name,
                time_folder,
            )
        elif (
            args.checkpoint_path == os.path.join(args.dataset_root_path, "ckpts")
            or args.checkpoint_path == "./ckpts"
        ):
            args.checkpoint_path = os.path.dirname(args.resume)
        elif (
            args.checkpoint_path != os.path.join(args.dataset_root_path, "ckpts")
            and args.checkpoint_path != "./ckpts"
            and args.resume != ""
        ):
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
