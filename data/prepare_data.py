import json
import os
import pickle

import numpy as np
import torch
import cv2
import ast

from data.custom_dataset import (
    DrivingDecisionDataset,
    PedestrianIntentDataset,
    MultiEpochsDataLoader,
)
from data.process_sequence import generate_data_sequence
from yolo_tracking.boxmot.utils import ROOT


def get_dataloader(args, shuffle_train=True, drop_last_train=True):
    task = args.task_name.split("_")[1]

    with open(
        os.path.join(args.database_path, f"{task}_database_train.pkl"), "rb"
    ) as fid:
        imdb_train = pickle.load(fid)
    train_seq = generate_data_sequence("train", imdb_train, args)
    with open(
        os.path.join(args.database_path, f"{task}_database_val.pkl"), "rb"
    ) as fid:
        imdb_val = pickle.load(fid)
    val_seq = generate_data_sequence("val", imdb_val, args)
    with open(
        os.path.join(args.database_path, f"{task}_database_test.pkl"), "rb"
    ) as fid:
        imdb_test = pickle.load(fid)
    test_seq = generate_data_sequence("test", imdb_test, args)

    train_d = get_train_val_data(
        train_seq, args, overlap=args.seq_overlap_rate
    )  # returned tracks
    val_d = get_train_val_data(val_seq, args, overlap=args.test_seq_overlap_rate)
    test_d = get_test_data(test_seq, args, overlap=args.test_seq_overlap_rate)

    # Create video dataset and dataloader
    match (args.task_name):
        case "driving_decision":
            train_dataset = DrivingDecisionDataset(train_d, args)
            val_dataset = DrivingDecisionDataset(val_d, args)
            test_dataset = DrivingDecisionDataset(test_d, args)
        case "ped_intent":
            train_dataset = PedestrianIntentDataset(train_d, args)
            val_dataset = PedestrianIntentDataset(val_d, args)
            test_dataset = PedestrianIntentDataset(test_d, args)
        case _:
            raise NotImplementedError
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    if not args.persist_dataloader:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            pin_memory=True,
            sampler=None,
            drop_last=drop_last_train,
            num_workers=4,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            num_workers=8,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            num_workers=4,
        )
    else:
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            pin_memory=True,
            sampler=None,
            drop_last=drop_last_train,
            num_workers=4,
            persistent_workers=True,
        )
        val_loader = MultiEpochsDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
        )
        test_loader = None
        if len(test_dataset) > 0:
            test_loader = MultiEpochsDataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                num_workers=4,
                persistent_workers=True,
            )

    return train_loader, val_loader, test_loader


def get_train_val_data(data, args, overlap=0.5):  # overlap==0.5, seq_len=15
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    print("Train/Val Tracks: ", tracks.keys())
    return tracks


def get_test_data(data, args, overlap=1):  # overlap==0.5, seq_len=15
    # return splited train/val dataset
    seq_len = args.max_track_size
    overlap = overlap
    tracks = get_tracks(data, seq_len, args.observe_length, overlap, args)
    print("Test Tracks: ", tracks.keys())
    return tracks


def get_tracks(data, seq_len, observed_seq_len, overlap, args):
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
        ]
    d = {}

    for k in d_types:
        d[k] = data[k]

    for k in d_types:
        # print(k, len(d[k]))
        # frame/bbox/intention_binary/reason_feats
        tracks = []
        for track_id in range(len(d[k])):
            track = d[k][track_id]
            # There are some sequences not adjacent
            frame_list = data["frame"][track_id]
            if len(frame_list) < args.max_track_size:  # 60:
                print(
                    "too few frames: ",
                    d["video_id"][track_id][0],
                    d["ped_id"][track_id][0],
                )
                continue
            splits = []
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
            sub_tracks = []
            for spl in splits:
                # explain the boundary:  end_idx - (15-1=14 gap) + cover last idx
                for i in range(spl[0], spl[1] - (seq_len - 1) + 1, overlap_stride):
                    sub_tracks.append(track[i : i + seq_len])
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

def consolidate_yolo_data(video_width, video_height):
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
    yolo_data_folder = os.path.join(ROOT, "runs", "track", "exp" ,"labels")
    file_list = os.listdir(yolo_data_folder)
    # Sort in numerical order of the frames
    sorted_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

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

                xtl = (x_center - width * 0.5) * video_width
                xbr = (x_center + width * 0.5) * video_width
                ytl = (y_center - height * 0.5) * video_height
                ybr = (y_center + height * 0.5) * video_height

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
    data_folder = os.path.join(ROOT, "yolo_results_data") 
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    file_count = 0
    for k, v in bbox_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 16 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "w") as f:
                    f.write(video_id + '\t' + k + '\t' + str(v[i: i+16]))
                file_count += 1
    file_count = 0
    for k, v in frames_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 16 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "a") as f:
                    f.write('\t' + str(v[i: i+16]))
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