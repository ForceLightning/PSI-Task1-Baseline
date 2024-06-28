from __future__ import annotations
import json
import os
import time
import pickle
from typing import Any

from data.process_sequence import T_drivingDB, T_intentDB
from utils.args import DefaultArguments
from .. import psi_scripts

"""
Database organization (Intent/Trajectory)

db = {
    'video_name': {
        'pedestrian_id': # track_id-#
        { 
            'frames': [0, 1, 2, ...], # target pedestrian appeared frames
            'cv_annotations': {
                'track_id': track_id, 
                'bbox': [xtl, ytl, xbr, ybr], 
            },
            'nlp_annotations': {
                vid_uid_pair: {'intent': [], 'description': [], 'key_frame': []},
                ...
            }
        }
    }
}

Database organisation (Driving Decision)
db = {
    - *video_name*: { # video name
        - 'frames': [0, 1, 2, ...], # list of frames that the target pedestrian appear
        - 'speed': [],
        - 'gps': [],
        - 'nlp_annotations': {
            - *annotator_id*: { # annotator's id/name
                - 'speed': [], # list of driving decision (speed) at speific frame, extended from key-frame annotations 
                - 'direction': [], # list of driving decision (direction) at speific frame, extended from key-frame annotations 
                - 'description': [], # list of explanation of the intent estimation for every frame from the current annotator_id
                - 'key_frame': [] # if the specific frame is key-frame, directly annotated by the annotator. 0-NOT key-frame, 1-key-frame
            },
            ...
        }
    }
}
"""


def create_database(args: DefaultArguments):
    for split_name in ["train", "val", "test"]:
        with open(args.video_splits) as f:
            datasplits = json.load(f)
        db_log = os.path.join(args.database_path, split_name + "_db_log.txt")
        with open(db_log, "w") as f:
            _ = f.write(f"Initialize {split_name} database \n")
            _ = f.write(time.strftime("%d%b%Y-%Hh%Mm%Ss") + "\n")
        # 1. Init db
        db = init_db(sorted(datasplits[split_name]), db_log, args)
        # 2. get intent, remove missing frames
        if args.task_name != "driving_decision":
            update_db_annotations(db, db_log, args)
        # 3. cut sequences, remove early frames before the first key frame, and after last key frame
        # cut_sequence(db, db_log, args)

        task = args.task_name.split("_")[1]
        database_name = f"{task}_database_" + split_name + ".pkl"
        with open(os.path.join(args.database_path, database_name), "wb") as fid:
            pickle.dump(db, fid)

    print("Finished collecting database!")


def add_ped_case(db, video_name, ped_name, nlp_vid_uid_pairs):
    if video_name not in db:
        db[video_name] = {}

    db[video_name][ped_name] = {  # ped_name is 'track_id' in cv-annotation
        "frames": None,  # [] list of frame_idx of the target pedestrian appear
        "cv_annotations": {
            "track_id": ped_name,
            "bbox": [],  # [] list of bboxes, each bbox is [xtl, ytl, xbr, ybr]
        },
        "nlp_annotations": {
            # [vid_uid_pair: {'intent': [], 'description': [], 'key_frame': []}]
        },
    }
    for vid_uid in nlp_vid_uid_pairs:
        db[video_name][ped_name]["nlp_annotations"][vid_uid] = {
            "intent": [],
            "description": [],
            "key_frame": [],
            # 0: not key frame (expanded from key frames with NLP annotations)
            # 1: key frame (labeled by NLP annotations)
        }


def add_case(
    db: dict[str, Any], video_name: str, cog_annotation, cv_annotation, db_log
):
    if video_name not in db:
        db[video_name] = {}

        # cog_annotation = annotation['pedestrians']['cognitive_annotations']
        # nlp_vid_uid_pairs = cog_annotation.keys()
    frame_list = list(cog_annotation["frames"].keys())
    frames = [int(f.split("_")[1]) for f in frame_list]

    db[video_name] = {  # ped_name is 'track_id' in cv-annotation
        "frames": frames,  # [] list of frame_idx of the target pedestrian appear
        "nlp_annotations": {
            # [vid_uid_pair: {'speed': [], 'direction': [], 'description': [], 'key_frame': []}]
        },
        "speed": [],
        "gps": [],
    }

    nlp_vid_uid_pairs = list(
        cog_annotation["frames"]["frame_0"]["cognitive_annotation"].keys()
    )
    for vid_uid in nlp_vid_uid_pairs:
        db[video_name]["nlp_annotations"][vid_uid] = {
            "driving_speed": [],
            "driving_direction": [],
            "description": [],
            "key_frame": [],
            # 0: not key frame (expanded from key frames with NLP annotations)
            # 1: key frame (labeled by NLP annotations)
        }

    first_ann_idx = len(frame_list) - 1
    last_ann_idx = -1
    for i in range(len(frame_list)):
        fname = frame_list[i]
        for vid_uid in nlp_vid_uid_pairs:
            db[video_name]["nlp_annotations"][vid_uid]["driving_speed"].append(
                cog_annotation["frames"][fname]["cognitive_annotation"][vid_uid][
                    "driving_decision_speed"
                ]
            )
            db[video_name]["nlp_annotations"][vid_uid]["driving_direction"].append(
                cog_annotation["frames"][fname]["cognitive_annotation"][vid_uid][
                    "driving_decision_direction"
                ]
            )
            db[video_name]["nlp_annotations"][vid_uid]["description"].append(
                cog_annotation["frames"][fname]["cognitive_annotation"][vid_uid][
                    "explanation"
                ]
            )
            db[video_name]["nlp_annotations"][vid_uid]["key_frame"].append(
                cog_annotation["frames"][fname]["cognitive_annotation"][vid_uid][
                    "key_frame"
                ]
            )
            # record first/last ann frame idx
            if (
                cog_annotation["frames"][fname]["cognitive_annotation"][vid_uid][
                    "key_frame"
                ]
                == 1
            ):
                first_ann_idx = min(first_ann_idx, i)
                last_ann_idx = max(last_ann_idx, i)
        try:
            db[video_name]["speed"].append(
                float(cv_annotation["frames"][fname]["speed(km/hr)"])
            )
            db[video_name]["gps"].append(cv_annotation["frames"][fname]["gps"])
        except:
            with open(db_log, "a") as f:
                f.write(f"NO speed and gps information:  {video_name} frame {fname} \n")

    # Cut sequences, only keep frames containing both driving decision & explanations
    db[video_name]["frames"] = db[video_name]["frames"][
        first_ann_idx : last_ann_idx + 1
    ]
    db[video_name]["speed"] = db[video_name]["speed"][first_ann_idx : last_ann_idx + 1]
    db[video_name]["gps"] = db[video_name]["gps"][first_ann_idx : last_ann_idx + 1]
    for vid_uid in nlp_vid_uid_pairs:
        db[video_name]["nlp_annotations"][vid_uid]["driving_speed"] = db[video_name][
            "nlp_annotations"
        ][vid_uid]["driving_speed"][first_ann_idx : last_ann_idx + 1]
        db[video_name]["nlp_annotations"][vid_uid]["driving_direction"] = db[
            video_name
        ]["nlp_annotations"][vid_uid]["driving_direction"][
            first_ann_idx : last_ann_idx + 1
        ]
        db[video_name]["nlp_annotations"][vid_uid]["description"] = db[video_name][
            "nlp_annotations"
        ][vid_uid]["description"][first_ann_idx : last_ann_idx + 1]
        db[video_name]["nlp_annotations"][vid_uid]["key_frame"] = db[video_name][
            "nlp_annotations"
        ][vid_uid]["key_frame"][first_ann_idx : last_ann_idx + 1]


def init_db(
    video_list: list[str], db_log: str, args: DefaultArguments
) -> T_intentDB | T_drivingDB:
    db: T_intentDB | T_drivingDB = {}
    # data_split = 'train' # 'train', 'val', 'test'
    dataroot = args.dataset_root_path
    # key_frame_folder = 'cognitive_annotation_key_frame'
    if args.dataset == "PSI2.0":
        extended_folder = "PSI2.0_TrainVal/annotations/cognitive_annotation_extended"
    elif args.dataset == "PSI1.0":
        extended_folder = "PSI1.0/annotations/cognitive_annotation_extended"
    else:
        raise NotImplementedError("Dataset not supported")

    for video_name in sorted(video_list):
        if args.task_name == "driving_decision":
            try:
                with open(
                    os.path.join(
                        dataroot, extended_folder, video_name, "driving_decision.json"
                    ),
                    "r",
                ) as f:
                    cog_annotation = json.load(f)
            except:
                with open(db_log, "a") as f:
                    _ = f.write(
                        f"Error loading {video_name} driving decision annotation json \n"
                    )
                continue

            if args.dataset == "PSI2.0":
                cv_folder = "PSI2.0_TrainVal/annotations/cv_annotation"
            elif args.dataset == "PSI1.0":
                cv_folder = "PSI1.0/annotations/cv_annotation"

            try:
                with open(
                    os.path.join(dataroot, cv_folder, video_name, "cv_annotation.json"),
                    "r",
                ) as f:
                    cv_annotation: (
                        psi_scripts.cv_annotation_schema.T_PSICVAnnDatabase
                    ) = json.load(f)
            except:
                with open(db_log, "a") as f:
                    _ = f.write(f"Error loading {video_name} cv annotation json \n")
                continue

            db[video_name] = {}

            add_case(db, video_name, cog_annotation, cv_annotation, db_log)

        else:  # intent and trajectory
            try:
                with open(
                    os.path.join(
                        dataroot, extended_folder, video_name, "pedestrian_intent.json"
                    ),
                    "r",
                ) as f:
                    annotation: (
                        psi_scripts.cv_annotation_schema.T_ExtendedIntentAnnoSchema
                    ) = json.load(f)
            except:
                with open(db_log, "a") as f:
                    _ = f.write(
                        f"Error loading {video_name} pedestrian intent annotation json \n"
                    )
                continue
            db[video_name] = {}
            for ped in annotation["pedestrians"].keys():
                cog_annotation = annotation["pedestrians"][ped]["cognitive_annotations"]
                nlp_vid_uid_pairs = cog_annotation.keys()
                add_ped_case(db, video_name, ped, nlp_vid_uid_pairs)
    return db


def split_frame_lists(
    frame_list: list[int],
    bbox_list: list[list[float]],
    pose_list: list[list[tuple[float, float]]],
    pose_mask: list[list[bool]],
    threshold: int = 60,
) -> tuple[
    list[list[int]],
    list[list[list[float]]],
    list[list[int]],
    list[list[list[tuple[float, float]]]],
    list[list[list[bool]]],
]:
    # For a sequence of an observed pedestrian, split into slices based on missing
    # frames
    frame_res: list[list[int]] = []
    bbox_res: list[list[list[float]]] = []
    inds_res: list[list[int]] = []
    pose_res: list[list[list[tuple[float, float]]]] = []
    pose_mask_res: list[list[list[bool]]] = []

    frame_split: list[int] = [frame_list[0]]  # frame list
    bbox_split: list[list[float]] = [bbox_list[0]]  # bbox list
    inds_split: list[int] = [0]
    pose_split: list[list[tuple[float, float]]] = [pose_list[0]]
    pose_mask_split: list[list[bool]] = [pose_mask[0]]

    for i in range(1, len(frame_list)):
        if frame_list[i] - frame_list[i - 1] == 1:  # consistent
            inds_split.append(i)
            frame_split.append(frame_list[i])
            bbox_split.append(bbox_list[i])
            pose_split.append(pose_list[i])
            pose_mask_split.append(pose_mask[i])
        else:  # # next position frame is missing observed
            if (
                len(frame_split) > threshold
            ):  # only take the slices longer than threshold=max_track_length=60
                inds_res.append(inds_split)
                frame_res.append(frame_split)
                bbox_res.append(bbox_split)
                pose_res.append(pose_split)
                pose_mask_res.append(pose_mask_split)
                inds_split = []
                frame_split = []
                bbox_split = []
                pose_split = []
                pose_mask_split = []
            else:  # ignore splits that are too short
                inds_split = []
                frame_split = []
                bbox_split = []
                pose_split = []
                pose_mask_split = []
    # break loop when i reaches the end of list
    if len(frame_split) > threshold:  # reach the end
        inds_res.append(inds_split)
        frame_res.append(frame_split)
        bbox_res.append(bbox_split)
        pose_res.append(pose_split)
        pose_mask_res.append(pose_mask_split)

    return frame_res, bbox_res, inds_res, pose_res, pose_mask_res


def get_intent_des(
    db: T_intentDB,
    vname: str,
    pid: str,
    split_inds: list[int],
    cog_annt: dict[str, psi_scripts.cv_annotation_schema.T_CognitiveAnnotation],
):
    # split_inds: the list of indices of the intent_annotations for the current split of pid in vname
    for vid_uid in cog_annt.keys():
        intent_list = cog_annt[vid_uid]["intent"]
        description_list = cog_annt[vid_uid]["description"]
        key_frame_list = cog_annt[vid_uid]["key_frame"]

        nlp_vid_uid = vid_uid
        db[vname][pid]["nlp_annotations"][nlp_vid_uid]["intent"] = [
            intent_list[i] for i in split_inds
        ]
        db[vname][pid]["nlp_annotations"][nlp_vid_uid]["description"] = [
            description_list[i] for i in split_inds
        ]
        db[vname][pid]["nlp_annotations"][nlp_vid_uid]["key_frame"] = [
            key_frame_list[i] for i in split_inds
        ]


def update_db_annotations(
    db: T_intentDB,
    db_log: str,
    args: DefaultArguments,
):
    dataroot = args.dataset_root_path
    # key_frame_folder = 'cognitive_annotation_key_frame'
    if args.dataset == "PSI2.0":
        extended_folder = "PSI2.0_TrainVal/annotations/cognitive_annotation_extended"
    elif args.dataset == "PSI1.0":
        extended_folder = "PSI1.0/annotations/cognitive_annotation_extended"
    else:
        raise NotImplementedError("Database not supported")

    video_list = sorted(db.keys())
    for video_name in video_list:
        ped_list: list[str] = list(db[video_name].keys())
        tracks: list[str] = list(db[video_name].keys())
        try:
            with open(
                os.path.join(
                    dataroot, extended_folder, video_name, "pedestrian_intent.json"
                ),
                "r",
            ) as f:
                annotation: (
                    psi_scripts.cv_annotation_schema.T_ExtendedIntentAnnoSchema
                ) = json.load(f)
        except:
            with open(db_log, "a") as f:
                _ = f.write(
                    f"Error loading {video_name} pedestrian intent annotation json \n"
                )
            continue

        for ped_id in ped_list:
            ped_track = annotation["pedestrians"][ped_id]
            observed_frames = ped_track["observed_frames"]
            observed_bboxes = ped_track["cv_annotations"]["bboxes"]
            observed_poses = ped_track["cv_annotations"]["skeleton"]
            observed_pose_masks = ped_track["cv_annotations"]["observed_skeleton"]
            cog_annotation = ped_track["cognitive_annotations"]
            if (
                len(observed_frames) == observed_frames[-1] - observed_frames[0] + 1
            ):  # no missing frames
                threshold = (
                    args.max_track_size
                )  # 16 for intent/driving decision; 60 for trajectory
                if len(observed_frames) > threshold:
                    cv_frame_list = observed_frames
                    cv_frame_box = observed_bboxes
                    db_track = db[video_name][ped_id]
                    db_track["frames"] = cv_frame_list
                    db_track["cv_annotations"]["bbox"] = cv_frame_box
                    db_track["cv_annotations"]["skeleton"] = observed_poses
                    db_track["cv_annotations"][
                        "observed_skeleton"
                    ] = observed_pose_masks
                    get_intent_des(
                        db,
                        video_name,
                        ped_id,
                        [*range(len(observed_frames))],
                        cog_annotation,
                    )
                else:  # too few frames observed
                    # print("Single ped occurs too short.", video_name, pedId, len(observed_frames))
                    with open(db_log, "a") as f:
                        _ = f.write(
                            f"Single ped occurs too short. {video_name}, {ped_id}, {len(observed_frames)} \n"
                        )
                    del db[video_name][ped_id]
            else:  # missing frames exist
                with open(db_log, "a") as f:
                    _ = f.write(
                        f"missing frames bbox noticed! , {video_name}, {ped_id}, {len(observed_frames)}, frames observed from , {observed_frames[-1] - observed_frames[0] + 1} \n"
                    )
                threshold = args.max_track_size  # 60
                (
                    cv_frame_list,
                    cv_frame_box,
                    cv_split_inds,
                    cv_frame_pose,
                    cv_frame_pose_mask,
                ) = split_frame_lists(
                    observed_frames,
                    observed_bboxes,
                    observed_poses,
                    observed_pose_masks,
                    threshold,
                )
                if len(cv_split_inds) == 0:
                    with open(db_log, "a") as f:
                        _ = f.write(
                            f"{video_name}, {ped_id}, After removing missing frames, no split left! \n"
                        )

                    del db[video_name][ped_id]
                elif len(cv_split_inds) == 1:
                    db[video_name][ped_id]["frames"] = cv_frame_list[0]
                    db[video_name][ped_id]["cv_annotations"]["bbox"] = cv_frame_box[0]
                    db[video_name][ped_id]["cv_annotations"]["skeleton"] = (
                        cv_frame_pose[0]
                    )
                    db[video_name][ped_id]["cv_annotations"]["observed_skeleton"] = (
                        cv_frame_pose_mask[0]
                    )

                    get_intent_des(
                        db, video_name, ped_id, cv_split_inds[0], cog_annotation
                    )
                else:
                    # multiple splits left after removing missing box frames
                    with open(db_log, "a") as f:
                        _ = f.write(
                            f"{len(cv_frame_list)} splits: , {[len(s) for s in cv_frame_list]} \n"
                        )
                    nlp_vid_uid_pairs = db[video_name][ped_id]["nlp_annotations"].keys()
                    for i in range(len(cv_frame_list)):
                        ped_splitId = ped_id + "-" + str(i)
                        add_ped_case(db, video_name, ped_splitId, nlp_vid_uid_pairs)
                        db[video_name][ped_splitId]["frames"] = cv_frame_list[i]
                        db[video_name][ped_splitId]["cv_annotations"]["bbox"] = (
                            cv_frame_box[i]
                        )
                        db[video_name][ped_splitId]["cv_annotations"]["skeleton"] = (
                            cv_frame_pose[i]
                        )
                        db[video_name][ped_splitId]["cv_annotations"][
                            "observed_skeleton"
                        ] = cv_frame_pose_mask[i]
                        get_intent_des(
                            db,
                            video_name,
                            ped_splitId,
                            cv_split_inds[i],
                            cog_annotation,
                        )
                        if (
                            len(
                                db[video_name][ped_splitId]["nlp_annotations"][
                                    list(
                                        db[video_name][ped_splitId][
                                            "nlp_annotations"
                                        ].keys()
                                    )[0]
                                ]["intent"]
                            )
                            == 0
                        ):
                            raise Exception("ERROR!")
                    del db[video_name][
                        ped_id
                    ]  # no pedestrian list left, remove this video
            tracks.remove(ped_id)
        if (
            len(db[video_name].keys()) < 1
        ):  # has no valid ped sequence! Remove this video!")
            with open(db_log, "a") as f:
                _ = f.write(
                    f"!!!!! Video, {video_name}, has no valid ped sequence! Remove this video! \n"
                )
            del db[video_name]
        if len(tracks) > 0:
            with open(db_log, "a") as f:
                _ = f.write(
                    f"{video_name} missing pedestrians annotations: {tracks}  \n"
                )


# def cut_sequence(db, db_log, args):
#     # only wanna use some of the sequence, thus cut edges
#     for vname in sorted(db.keys()):
#         for pid in sorted(db[vname].keys()):
#             frames = db[vname][pid]['frames']
#             first_cog_idx = len(frames) + 1
#             last_cog_idx = -1
#             for uv in db[vname][pid]['nlp_annotations'].keys():
#                 key_frame_list = db[vname][pid]['nlp_annotations'][uv]['key_frame']
#                 for i in range(len(key_frame_list)):
#                     if key_frame_list[i] == 1:
#                         first_cog_idx = min(first_cog_idx, i)
#                         break
#                 for j in range(len(key_frame_list)-1, -1, -1):
#                     if key_frame_list[j] == 1:
#                         last_cog_idx = max(last_cog_idx, j)
#                         break
#
#             if first_cog_idx > len(frames) or last_cog_idx < 0:
#                 print("ERROR! NO key frames found in ", vname, pid)
#             else:
#                 print("First and last annotated key frames are: ", frames[first_cog_idx], frames[last_cog_idx])
#                 print('In total frames # = ', last_cog_idx - first_cog_idx, ' out of ', len(frames))
#
#             if last_cog_idx - first_cog_idx < args.max_track_size:
#                 print(vname, pid, " too few frames left after cutting sequence.")
#                 del db[vname][pid]
#             else:
#                 db[vname][pid]['frames'] = db[vname][pid]['frames'][first_cog_idx: last_cog_idx+1]
#                 db[vname][pid]['cv_annotations']['bbox'] = db[vname][pid]['cv_annotations']['bbox'][first_cog_idx: last_cog_idx + 1]
#                 db[vname][pid]['frames'] = db[vname][pid]['frames'][first_cog_idx: last_cog_idx + 1]
#                 for uv in db[vname][pid]['nlp_annotations'].keys():
#                     db[vname][pid]['nlp_annotations'][uv]['intent'] = db[vname][pid]['nlp_annotations'][uv]['intent'][first_cog_idx: last_cog_idx + 1]
#                     db[vname][pid]['nlp_annotations'][uv]['description'] = db[vname][pid]['nlp_annotations'][uv]['description'][
#                                                           first_cog_idx: last_cog_idx + 1]
#                     db[vname][pid]['nlp_annotations'][uv]['key_frame'] = db[vname][pid]['nlp_annotations'][uv]['key_frame'][
#                                                                       first_cog_idx: last_cog_idx + 1]
#         if len(db[vname].keys()) < 1:
#             print(vname, "After cutting sequence edges, not enough frames left! Delete!")
#             del db[vname]
