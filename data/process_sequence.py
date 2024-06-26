import collections
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy import typing as npt

from utils.args import DefaultArguments

T_intentDB: TypeAlias = dict[
    str,  # video name
    dict[
        str,  # pedestrian id
        dict[
            {
                "frames": list[int],
                "cv_annotations": dict[
                    {
                        "track_id": str,
                        "bbox": list[list[float]],
                    }
                ],
                "nlp_annotations": dict[
                    str,  # annotator id
                    dict[
                        {
                            "intent": list[Literal["not_sure", "cross", "not_cross"]],
                            "description": list[str],
                            "key_frame": list[int],
                        }
                    ],
                ],
            }
        ],
    ],
]

T_drivingDB: TypeAlias = dict[
    str,  # video name
    dict[
        {
            "frames": list[int],
            "speed": list[float],
            "gps": list[tuple[str, str]],
            "nlp_annotations": dict[
                str,  # annotator id
                dict[
                    {
                        "driving_speed": list[
                            Literal["increaseSpeed", "decreaseSpeed", "maintainSpeed"]
                        ],
                        "driving_direction": list[
                            Literal["goStraight", "turnLeft", "turnRight"]
                        ],
                        "description": list[str],
                        "key_frame": list[int],
                    }
                ],
            ],
        }
    ],
]

T_intentSeq: TypeAlias = dict[
    {
        "frame": list[list[int]],
        "bbox": list[list[list[float]]],
        "intention_prob": list[list[float | np.float_]],
        "intention_binary": list[list[int]],
        "ped_id": list[list[str]],
        "video_id": list[list[str]],
        "disagree_score": list[list[float | np.float_]],
        "description": list[list[list[str]]],
    }
]

T_intentSeqNumpy: TypeAlias = dict[
    {
        "frame": npt.NDArray[np.int_],
        "bbox": npt.NDArray[np.float_],
        "intention_prob": npt.NDArray[np.float_],
        "ped_id": npt.NDArray[np.str_],
        "video_id": npt.NDArray[np.str_],
        "disagree_score": npt.NDArray[np.float_],
        "description": npt.NDArray[Any],
    }
]

T_drivingSeq: TypeAlias = dict[
    {
        "frame": list[list[int]],
        "video_id": list[list[str]],
        "speed": list[list[float]],
        "gps": list[list[str]],
        "driving_speed": list[list[int]],
        "driving_speed_prob": list[list[float]],
        "driving_direction": list[list[int]],
        "driving_direction_prob": list[list[float]],
        "description": list[list[list[str]]],
    }
]

T_drivingSeqNumpy: TypeAlias = dict[
    {
        "frame": npt.NDArray[np.int_],
        "video_id": npt.NDArray[np.str_],
        "speed": npt.NDArray[np.float_],
        "gps": npt.NDArray[np.str_],
        "driving_speed": npt.NDArray[np.int_],
        "driving_speed_prob": npt.NDArray[np.float_],
        "driving_direction": npt.NDArray[np.int_],
        "driving_direction_prob": npt.NDArray[np.float_],
        "description": npt.NDArray[Any],
    }
]


def generate_data_sequence(
    set_name: str,
    database: T_intentDB | T_drivingDB,
    args: DefaultArguments,
) -> T_intentSeq | T_drivingSeq:
    frame_seq: list[list[int]] = []
    video_seq: list[list[str]] = []
    description_seq: list[list[list[str]]] = []
    if args.task_name == "driving_decision":
        speed_seq: list[list[float]] = []
        gps_seq: list[list[str]] = []
        driving_speed_seq: list[list[int]] = []
        driving_direction_seq: list[list[int]] = []
        driving_speed_prob_seq: list[list[float]] = []
        driving_direction_prob_seq: list[list[float]] = []

        video_ids = sorted(database.keys())
        for video in sorted(video_ids):  # video_name: e.g., 'video_0001'
            frame_seq.append(database[video]["frames"])
            n = len(database[video]["frames"])
            video_seq.append([video] * n)
            speed_seq.append(database[video]["speed"])
            gps_seq.append(database[video]["gps"])

            dr_speed, dr_dir, dr_speed_prob, dr_dir_prob, descrp = get_driving(
                database, video, args
            )
            driving_speed_seq.append(dr_speed)
            driving_direction_seq.append(dr_dir)
            driving_speed_prob_seq.append(dr_speed_prob)
            driving_direction_prob_seq.append(dr_dir_prob)
            description_seq.append(descrp)

        return {
            "frame": frame_seq,
            "video_id": video_seq,
            "speed": speed_seq,
            "gps": gps_seq,
            "driving_speed": driving_speed_seq,
            "driving_speed_prob": driving_speed_prob_seq,
            "driving_direction": driving_direction_seq,
            "driving_direction_prob": driving_direction_prob_seq,
            "description": description_seq,
        }

    intention_prob: list[list[float | np.float_]] = []
    intention_binary: list[list[int]] = []
    pids_seq: list[list[str]] = []
    box_seq: list[list[list[float]]] = []
    disagree_score_seq: list[list[float | np.float_]] = []

    video_ids = sorted(database.keys())
    for video in sorted(video_ids):  # video_name: e.g., 'video_0001'
        for ped in sorted(database[video].keys()):  # ped_id: e.g., 'track_1'
            frame_seq.append(database[video][ped]["frames"])
            box_seq.append(database[video][ped]["cv_annotations"]["bbox"])

            n = len(database[video][ped]["frames"])
            pids_seq.append([ped] * n)
            video_seq.append([video] * n)
            intents, probs, disgrs, descripts = get_intent(database, video, ped, args)
            intention_prob.append(probs)
            intention_binary.append(intents)
            disagree_score_seq.append(disgrs)
            description_seq.append(descripts)

    return {
        "frame": frame_seq,
        "bbox": box_seq,
        "intention_prob": intention_prob,
        "intention_binary": intention_binary,
        "ped_id": pids_seq,
        "video_id": video_seq,
        "disagree_score": disagree_score_seq,
        "description": description_seq,
    }


def get_intent(
    database: T_intentDB, video_name: str, ped_id: str, args: DefaultArguments
):
    prob_seq: list[float | np.float_] = []
    intent_seq: list[int] = []
    disagree_seq: list[float | np.float_] = []
    description_seq: list[list[str]] = []
    n_frames = len(database[video_name][ped_id]["frames"])

    if args.intent_type in ("major", "soft_vote"):
        vid_uid_pairs = sorted((database[video_name][ped_id]["nlp_annotations"].keys()))
        n_users = len(vid_uid_pairs)
        for i in range(n_frames):
            labels = [
                database[video_name][ped_id]["nlp_annotations"][vid_uid]["intent"][i]
                for vid_uid in vid_uid_pairs
            ]
            descriptions = [
                database[video_name][ped_id]["nlp_annotations"][vid_uid]["description"][
                    i
                ]
                for vid_uid in vid_uid_pairs
            ]

            if args.intent_num == 3:  # major 3 class, use cross-entropy loss
                uni_lbls, uni_cnts = np.unique(labels, return_counts=True)
                intent_binary = uni_lbls[np.argmax(uni_cnts)]
                if intent_binary == "not_cross":
                    intent_binary = 0
                elif intent_binary == "not_sure":
                    intent_binary = 1
                elif intent_binary == "cross":
                    intent_binary = 2
                else:
                    raise ValueError(
                        "ERROR intent label from database: ", intent_binary
                    )

                intent_prob = np.max(uni_cnts) / n_users
                prob_seq.append(intent_prob)
                intent_seq.append(intent_binary)
                disagree_seq.append(1 - intent_prob)
                description_seq.append(descriptions)
            elif (
                args.intent_num == 2
            ):  # only counts labels not "not-sure", but will involve issues if all annotators are not-sure.
                raise ValueError("Sequence processing not implemented!")
            else:
                pass
    elif args.intent_type == "mean":
        vid_uid_pairs = sorted((database[video_name][ped_id]["nlp_annotations"].keys()))
        n_users = len(vid_uid_pairs)
        for i in range(n_frames):
            labels = [
                database[video_name][ped_id]["nlp_annotations"][vid_uid]["intent"][i]
                for vid_uid in vid_uid_pairs
            ]
            descriptions = [
                database[video_name][ped_id]["nlp_annotations"][vid_uid]["description"][
                    i
                ]
                for vid_uid in vid_uid_pairs
            ]

            if args.intent_num == 2:
                for j, label in enumerate(labels):
                    if label == "not_sure":
                        labels[j] = 0.5
                    elif label == "not_cross":
                        labels[j] = 0
                    elif label == "cross":
                        labels[j] = 1
                    else:
                        raise ValueError("Unknown intent label: ", label)
                # [0, 0.5, 1]
                intent_prob = np.mean(labels)
                intent_binary = 0 if intent_prob < 0.5 else 1
                prob_seq.append(intent_prob)
                intent_seq.append(intent_binary)
                disagree_score = (
                    sum([1 if lbl != intent_binary else 0 for lbl in labels]) / n_users
                )
                disagree_seq.append(disagree_score)
                description_seq.append(descriptions)

    return intent_seq, prob_seq, disagree_seq, description_seq


def get_driving(database: T_drivingDB, video: str, args: DefaultArguments):
    # driving_speed, driving_dir, dr_speed_dsagr, dr_dir_dsagr, description
    n = len(database[video]["frames"])
    dr_speed: list[int] = []
    dr_dir: list[int] = []
    dr_speed_prob: list[float] = []
    dr_dir_prob: list[float] = []
    description: list[list[str]] = []
    nlp_vid_uid_pairs = list(database[video]["nlp_annotations"].keys())
    for i in range(n):
        speed_ann, speed_prob = get_driving_speed_to_category(database, video, i)
        dir_ann, dir_prob = get_driving_direction_to_category(database, video, i)
        des_ann = [
            database[video]["nlp_annotations"][vu]["description"][i]
            for vu in nlp_vid_uid_pairs
            if database[video]["nlp_annotations"][vu]["description"][i] != ""
        ]
        dr_speed.append(speed_ann)
        dr_speed_prob.append(speed_prob)
        dr_dir.append(dir_ann)
        dr_dir_prob.append(dir_prob)
        description.append(
            des_ann
        )  # may contains different number of descriptions for different frames

    return dr_speed, dr_dir, dr_speed_prob, dr_dir_prob, description


def get_driving_speed_to_category(
    database: dict[str, Any], video: str, i: int
) -> tuple[int, float]:
    nlp_vid_uid_pairs: list[str] = list(database[video]["nlp_annotations"].keys())
    speed_ann_list = [
        database[video]["nlp_annotations"][vu]["driving_speed"][i]
        for vu in nlp_vid_uid_pairs
        if database[video]["nlp_annotations"][vu]["driving_speed"][i] != ""
    ]
    counter = collections.Counter(speed_ann_list)
    most_common = counter.most_common(1)[0]
    speed_ann = str(most_common[0])
    speed_prob = int(most_common[1]) / len(speed_ann_list)
    # speed_ann = max(set(speed_ann_list), key=speed_ann_list.count)

    match speed_ann:
        case "maintainSpeed":
            speed_ann = 0
        case "decreaseSpeed":
            speed_ann = 1
        case "increaseSpeed":
            speed_ann = 2
        case _:
            raise ValueError("Unknown driving speed annotation: " + str(most_common))
    return speed_ann, speed_prob


def get_driving_direction_to_category(database: T_drivingDB, video: str, i: int):
    nlp_vid_uid_pairs: list[str] = list(database[video]["nlp_annotations"].keys())
    direction_ann_list = [
        database[video]["nlp_annotations"][vu]["driving_direction"][i]
        for vu in nlp_vid_uid_pairs
        if database[video]["nlp_annotations"][vu]["driving_direction"][i] != ""
    ]
    counter = collections.Counter(direction_ann_list)
    most_common = counter.most_common(1)[0]
    direction_ann = str(most_common[0])
    direction_prob = int(most_common[1]) / len(direction_ann_list)
    # direction_ann = max(set(direction_ann_list), key=direction_ann_list.count)

    if direction_ann == "goStraight":
        direction_ann = 0
    elif direction_ann == "turnLeft":
        direction_ann = 1
    elif direction_ann == "turnRight":
        direction_ann = 2
    else:
        raise ValueError("Unknown driving direction annotation: " + direction_ann)

    return direction_ann, direction_prob
