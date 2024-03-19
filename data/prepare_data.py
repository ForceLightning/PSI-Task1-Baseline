from copy import deepcopy
import os
import pickle
from typing import Any

import numpy as np
from numpy import typing as npt
from torch.utils.data import DataLoader
import cv2

from data.custom_dataset import (
    DrivingDecisionDataset,
    MultiEpochsDataLoader,
    PedestrianIntentDataset,
)
from data.process_sequence import (
    T_drivingSeq,
    T_drivingSeqNumpy,
    T_intentSeq,
    T_intentSeqNumpy,
    generate_data_sequence,
)
from utils.args import DefaultArguments
from yolo_tracking.boxmot.utils import ROOT


def get_dataloader(
    args: DefaultArguments,
    shuffle_train: bool = True,
    drop_last_train: bool = True,
    load_test: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any] | None]:
    """Instantiatees dataloaders based on the prediction task.

    :param DefaultArguments args: Training arguments
    :param bool shuffle_train: Whether to shuffle the training dataloader, defaults to
        True.
    :param bool drop_last_train: Whether to drop the last training batch, defaults to
        True.
    :param bool load_test: Whether to load the test dataloader, defaults to False.
    :returns: Training, Validation, and Testing dataloaders.
    :rtype: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
    """
    task = args.task_name.split("_")[1]

    with open(
        os.path.join(
            args.dataset_root_path, args.database_path, f"{task}_database_train.pkl"
        ),
        "rb",
    ) as fid:
        imdb_train = pickle.load(fid)
    train_seq = generate_data_sequence("train", imdb_train, args)
    with open(
        os.path.join(
            args.dataset_root_path, args.database_path, f"{task}_database_val.pkl"
        ),
        "rb",
    ) as fid:
        imdb_val = pickle.load(fid)
    val_seq = generate_data_sequence("val", imdb_val, args)
    if load_test:
        with open(
            os.path.join(
                args.dataset_root_path, args.database_path, f"{task}_database_test.pkl"
            ),
            "rb",
        ) as fid:
            imdb_test = pickle.load(fid)
        test_seq = generate_data_sequence("test", imdb_test, args)

    train_d = get_train_val_data(
        train_seq, args, overlap=args.seq_overlap_rate
    )  # returned tracks
    val_d = get_train_val_data(val_seq, args, overlap=args.seq_overlap_rate)
    test_d = None
    if load_test:
        test_d = get_test_data(test_seq, args, overlap=args.test_seq_overlap_rate)

    # Create video dataset and dataloader
    match (args.task_name):
        case "driving_decision":
            train_dataset = DrivingDecisionDataset(train_d, args)
            val_dataset = DrivingDecisionDataset(val_d, args)
            test_dataset = (
                DrivingDecisionDataset(test_d, args) if load_test and test_d else None
            )
        case "ped_intent":
            train_dataset = PedestrianIntentDataset(train_d, args)
            val_dataset = PedestrianIntentDataset(val_d, args)
            test_dataset = (
                PedestrianIntentDataset(test_d, args) if load_test and test_d else None
            )
        case "ped_traj":
            train_dataset = PedestrianIntentDataset(train_d, args)
            val_dataset = PedestrianIntentDataset(val_d, args)
            test_dataset = (
                PedestrianIntentDataset(test_d, args) if load_test and test_d else None
            )
        case _:
            raise NotImplementedError

    print(
        len(train_dataset), len(val_dataset), len(test_dataset) if test_dataset else 0
    )

    if not args.persist_dataloader:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            # pin_memory=True,
            sampler=None,
            drop_last=drop_last_train,
            num_workers=8,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            # pin_memory=True,
            sampler=None,
            drop_last=False,
            num_workers=8,
        )
        test_loader = None
        if test_dataset and load_test:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                # pin_memory=True,
                sampler=None,
                drop_last=False,
                num_workers=8,
            )
    else:
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            # pin_memory=True,
            sampler=None,
            drop_last=drop_last_train,
            num_workers=8,
            persistent_workers=True,
        )
        val_loader = MultiEpochsDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            # shuffle=False,
            shuffle=shuffle_train,
            # pin_memory=True,
            sampler=None,
            drop_last=False,
            num_workers=8,
            persistent_workers=True,
        )
        test_loader = None
        if test_dataset and load_test:
            test_loader = MultiEpochsDataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                # pin_memory=True,
                sampler=None,
                drop_last=False,
                num_workers=8,
                persistent_workers=False,
            )

    return train_loader, val_loader, test_loader


def get_train_val_data(
    data: T_intentSeq | T_drivingSeq, args: DefaultArguments, overlap: float = 0.5
) -> T_intentSeqNumpy | T_drivingSeqNumpy:  # overlap==0.5, seq_len=15
    """Given the data sequences, return preprocessed dicts of numpy data.

    :param data: Data sequences to be processed.
    :type data: T_intentSeq or T_drivingSeq
    :param DefaultArguments args: Training arguments.
    :param float overlap: Amount of overlap between sequences. Defaults to 0.5.
    :returns: Processed data.
    :rtype: T_intentSeqNumpy or T_drivingSeqNumpy
    """
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    print("Train/Val Tracks: ", tracks.keys())
    return tracks


def get_test_data(
    data: T_intentSeq | T_drivingSeq, args: DefaultArguments, overlap: float = 1
) -> T_intentSeqNumpy | T_drivingSeqNumpy:
    """Given the data sequences, return preprocessed dicts of numpy data.

    :param data: Data sequences to be processed.
    :type data: T_intentSeq or T_drivingSeq
    :param DefaultArguments args: Training arguments.
    :param float overlap: Amount of overlap between sequences. Defaults to 0.5.
    :returns: Processed data.
    :rtype: T_intentSeqNumpy or T_drivingSeqNumpy
    """
    # return splited train/val dataset
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    print("Test Tracks: ", tracks.keys())
    return tracks


def get_tracks(
    data: T_intentSeq | T_drivingSeq,
    seq_len: int,
    observed_seq_len: int,
    overlap: float,
    args: DefaultArguments,
) -> T_intentSeqNumpy | T_drivingSeqNumpy:
    """Processes data sequences into tracks of overlapping data.

    :param data: Data sequences to be processed.
    :type data: T_intentSeq or T_drivingSeq
    :param int seq_length: Max sequence length.
    :param int observed_seq_length: Contextual sequence length.
    :param float overlap: How much tracks should overlap with each other.
    :param DefaultArguments args: Training arguments.
    :returns tracks: Processed data.
    :rtype: T_intentSeqNumpy or T_drivingSeqNumpy
    """
    overlap_stride = (
        observed_seq_len if overlap == 0 else int((1 - overlap) * observed_seq_len)
    )  # default: int(0.5*15) == 7

    overlap_stride = (
        1 if overlap_stride < 1 else overlap_stride
    )  # when test, overlap=1, stride=1

    if args.task_name == "driving_decision":
        d_types = [
            "video_id",
            "frame",
            "speed",
            "gps",
            "driving_speed",
            "driving_speed_prob",
            "driving_direction",
            "driving_direction_prob",
            "description",
        ]
    else:
        d_types = [
            "video_id",
            "ped_id",
            "frame",
            "bbox",
            "intention_binary",
            "intention_prob",
            "disagree_score",
            "description",
            "skeleton",
            "observed_skeleton",
        ]
    d = deepcopy(data)

    for k in d_types:
        # print(k, len(d[k]))
        # frame/bbox/intention_binary/reason_feats
        tracks: list[list[int | float | np.float_ | str | np.bool_]] = []
        for track_id, track in enumerate(d[k]):
            # There are some sequences not adjacent
            frame_list = data["frame"][track_id]
            if len(frame_list) < args.max_track_size:  # 60:
                # print(
                #     f"key: {k}, track_id: {track_id}, len(d['video_id']): {len(d['video_id'])}"
                # )
                # print(d["video_id"][track_id])
                print(
                    "too few frames: ",
                    d["video_id"][track_id][0],
                    d["ped_id"][track_id][0],
                )
                continue
            splits: list[list[int]] = []
            start = -1
            for fid in range(len(frame_list) - 1):
                if start == -1:
                    start = fid  # frame_list[f]
                if frame_list[fid] + 1 == frame_list[fid + 1]:
                    if fid + 1 == len(frame_list) - 1:
                        splits.append([start, fid + 1])
                    continue
                else:
                    # current f is the end of current piece
                    splits.append([start, fid])
                    start = -1
            if len(splits) != 1:
                print("NOT one missing split found: ", splits)
                raise NotImplementedError
            else:  # len(splits) == 1, No missing frames from the database.
                pass
            sub_tracks: list[int | float | np.float_ | str] = []
            for spl in splits:
                # explain the boundary:  end_idx - (15-1=14 gap) + cover last idx
                for i in range(spl[0], spl[1] - (seq_len - 1) + 1, overlap_stride):
                    sub_track = track[i : i + seq_len]
                    sub_tracks.append(sub_track)
            tracks.extend(sub_tracks)

        try:
            d[k] = np.array(tracks)
        except:
            d[k] = np.array(tracks, dtype=object)

    return d


def get_video_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()

    return width, height


def consolidate_yolo_data(image_width, image_height):
    bbox_holder = {}
    frames_holder = {}
    yolo_data_folder = os.path.join(ROOT, "runs", "track", "exp", "labels")
    file_list = os.listdir(yolo_data_folder)
    # Sort in numerical order of the frames
    sorted_list = sorted(file_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    for file in sorted_list:
        string = file.split("_")
        video_id = string[0] + "_" + string[1]
        frame_id = string[-1].split(".")[0]
        result_file = os.path.join(yolo_data_folder, file)
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                elements = line.split(" ")
                x_center = float(elements[1])
                y_center = float(elements[2])
                width = float(elements[3])
                height = float(elements[4])

                xtl = (x_center - width * 0.5) * image_width
                xbr = (x_center + width * 0.5) * image_width
                ytl = (y_center - height * 0.5) * image_height
                ybr = (y_center + height * 0.5) * image_height

                # To prevent error when YOLO gives iffy predictions
                try:
                    track_id = elements[5].strip()
                    track_id = "track_" + track_id
                    if track_id not in bbox_holder:
                        bbox_holder[track_id] = [[xtl, ytl, xbr, ybr]]
                    else:
                        bbox_holder[track_id].append([xtl, ytl, xbr, ybr])
                    if track_id not in frames_holder:
                        frames_holder[track_id] = [frame_id]
                    else:
                        frames_holder[track_id].append(frame_id)
                except:
                    pass

    return bbox_holder, frames_holder, video_id


# Make folder with the data then create dataset from there
def save_data_to_txt(bbox_dict, frames_dict, video_id):
    data_folder = os.path.join(ROOT, "yolo_results_data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    file_count = 0
    for k, v in bbox_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 16 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "w") as f:
                    f.write(video_id + "\t" + k + "\t" + str(v[i : i + 16]))
                file_count += 1
    file_count = 0
    for k, v in frames_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 16 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "a") as f:
                    f.write("\t" + str(v[i : i + 16]))
                file_count += 1
