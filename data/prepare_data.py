import os
import pickle
from copy import deepcopy
from typing import Any

import numpy as np
from numpy import typing as npt
from torch.utils.data import DataLoader
import cv2
import ast
import json

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


def get_dataloader(
    args: DefaultArguments,
    shuffle_train: bool = True,
    drop_last_train: bool = True,
    load_test: bool = False,
    yolo_transforms: bool = False,
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
            train_dataset = DrivingDecisionDataset(
                train_d, args, apply_yolo_transforms=yolo_transforms
            )
            val_dataset = DrivingDecisionDataset(
                val_d, args, apply_yolo_transforms=yolo_transforms
            )
            test_dataset = (
                DrivingDecisionDataset(
                    test_d, args, apply_yolo_transforms=yolo_transforms
                )
                if load_test and test_d
                else None
            )
        case "ped_intent":
            train_dataset = PedestrianIntentDataset(
                train_d, args, apply_yolo_transforms=yolo_transforms
            )
            val_dataset = PedestrianIntentDataset(
                val_d, args, apply_yolo_transforms=yolo_transforms
            )
            test_dataset = (
                PedestrianIntentDataset(
                    test_d, args, apply_yolo_transforms=yolo_transforms
                )
                if load_test and test_d
                else None
            )
        case "ped_traj":
            train_dataset = PedestrianIntentDataset(
                train_d, args, apply_yolo_transforms=yolo_transforms
            )
            val_dataset = PedestrianIntentDataset(
                val_d, args, apply_yolo_transforms=yolo_transforms
            )
            test_dataset = (
                PedestrianIntentDataset(
                    test_d, args, apply_yolo_transforms=yolo_transforms
                )
                if load_test and test_d
                else None
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
                persistent_workers=True,
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
    """
    Get height and width of the video

    Args:
    video_path : str
        Path to the video
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return None, None

    # Get the video frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()

    return width, height

def consolidate_yolo_data(video_id):
    """
    Consolidate the output data from YOLO

    Args:
    video_width : int
        Width of the video
    video_height: int
        Height of the height
    """
    bbox_holder = {}
    frames_holder = {}
    yolo_data = os.path.join(os.getcwd(), video_id + ".txt")

    with open(yolo_data, "r") as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split(" ")
            track_id = elements[0]
            x1 = float(elements[1])
            y1 = float(elements[2])
            x2 = float(elements[3])
            y2 = float(elements[4])
            frame_id = elements[-1].strip()

            if track_id not in bbox_holder:
                bbox_holder[track_id] = [[x1, y1, x2, y2]]
            else:
                bbox_holder[track_id].append([x1, y1, x2, y2])
            if track_id not in frames_holder:
                    frames_holder[track_id] = [frame_id]
            else:
                frames_holder[track_id].append(frame_id)
            # To prevent error when YOLO gives iffy predictions
            # try:
            #     track_id = elements[5].strip()
            #     track_id = "track_" + track_id
            #     if track_id not in bbox_holder:
            #         bbox_holder[track_id] = [[xtl, ytl, xbr, ybr]]
            #     else:
            #         bbox_holder[track_id].append([xtl, ytl, xbr, ybr])
            #     if track_id not in frames_holder:
            #         frames_holder[track_id] = [frame_id]
            #     else:
            #         frames_holder[track_id].append(frame_id)
            # except:
            #     pass
    
    return bbox_holder, frames_holder, video_id
 
def save_data_to_txt(bbox_dict, frames_dict, video_id):
    """
    Make folder with the data then create dataset from there

    Args:
    bbox_dict : dict
        Dictionary containing the ped_id and the bbox
    frames_dict: dict
        Dictionary containing the ped_id and the frames
    video_id : str
        The video id
    """
    data_folder = os.path.join(os.getcwd(), "yolo_results_data") 
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    file_count = 0
    for k, v in bbox_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 15 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "w") as f:
                    f.write(video_id + '\t' + k + '\t' + str(v[i: i+15]))
                file_count += 1
    file_count = 0
    for k, v in frames_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 15 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "a") as f:
                    f.write('\t' + str(v[i: i+15]))
                file_count += 1

def visualise_annotations(annotation_path, selected_frame):
    """
    Visualise the bounding boxes that are fed into TCN model for sanity check

    Args:
    annotation_path : str
        path to text with labeling for Intent Prediction
    selected_frame : int
        The frame to visualise (0-14)
    """
    # Read annotations
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    for ann in annotations:
        elements = ann.split("\t")
        vid_id = elements[0]
        ped_id = elements[1]
        bboxes = ast.literal_eval(elements[2])
        bbox = bboxes[selected_frame]
        xtl = int(bbox[0])
        ytl = int(bbox[1])
        xbr = int(bbox[2])
        ybr = int(bbox[3])
        frames = ast.literal_eval(elements[3])
        frame = str(frames[selected_frame]).zfill(3)

        # Load the image
        image_path = os.path.join(os.getcwd(), "frames", vid_id, frame + ".jpg")
        image = cv2.imread(image_path)

        # Draw bounding box
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)
        cv2.putText(img=image, 
                    text=ped_id,
                    org=(xbr, ybr), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale= 0.5,
                    color=(0, 0, 255)
                    )

    # Display the image
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualise_intent(annotation_path, intent_path):
    """
    Visualise the pedestrian intent (Bbox is 1 frame behind)

    Args:
    annotation_path: str
        path to text with yolo_annotation
    intent_path : str
        path to text with labeling for Intent Prediction
    video_width : int
        Width of the video
    video_height: int
        Height of the height    
    """

    with open(intent_path, 'r') as f:
        intent_results = f.read()

    # Parse JSON data
    parsed_data = json.loads(intent_results)

    # Loop through the JSON data and extract frame and intent for each track
    for vid_id, tracks in parsed_data.items():
        print("Video:", vid_id)
        for track, frames in tracks.items():
            print("Track:", track)
            for frame, details in frames.items():
                yolo_frame = int(frame) 
                with open(annotation_path, 'r') as f:
                    annotations = f.readlines()
                for ann in annotations:
                    element = ann.split(" ")

                    if ("track_" + element[0]) == track and element[-1].strip() == frame: 
                        xtl = float(element[1])
                        ytl = float(element[2])
                        xbr = float(element[3])
                        ybr = float(element[4])

                        frame = frame.zfill(3)
                        intent = details["intent"]
                        image_path = os.path.join(os.getcwd(), "frames", vid_id, frame + ".jpg")
                        image = cv2.imread(image_path)
                        print("GT Frame:", frame, "| Intent:", intent, "| Bbox Frame:", yolo_frame)
                        cv2.rectangle(image, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (255, 0, 0), 2)
                        cv2.putText(img=image, 
                        text=str(intent),
                        org=(int(xbr), int(ybr)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale= 1,
                        color=(0, 0, 255)
                        )
                        cv2.putText(img=image, 
                        text=track,
                        org=(int(xbr), int(ytl)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale= 1,
                        color=(0, 0, 255)
                        )
                        cv2.imshow('Annotated Image', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()