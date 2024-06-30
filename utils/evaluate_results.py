"""Evaluation functions for intent, trajectory, and driving prediction.
"""

from __future__ import annotations
import json
import os
from typing import Any, TypeAlias

import numpy as np
from numpy import typing as npt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
)

from utils.args import DefaultArguments

T_intentGT: TypeAlias = dict[
    str,  # Video ID
    dict[
        str,  # Pedestrian ID
        dict[
            str,  # Frame ID
            dict[{"intent": int, "intent_prob": float, "disagreement": float}],
        ],
    ],
]
"""Intent ground truth type alias.

A dictionary with the following structure:
{
    "Video ID": {
        "Pedestrian ID": {
            "Frame ID": {
                "intent": int,
                "intent_prob": float,
                "disagreement": float
            }
        }
    }
}
"""

T_intentPred: TypeAlias = dict[
    str,  # Video ID
    dict[
        str,  # Pedestrian ID
        dict[
            str,  # Frame ID
            dict[{"intent": int, "intent_prob": float}],
        ],
    ],
]
"""Intent prediction type alias.

A dictionary with the following structure:
{
    "Video ID": {
        "Pedestrian ID": {
            "Frame ID": {
                "intent": int,
                "intent_prob": float
            }
        }
    }
}
"""

T_intentPredGT: TypeAlias = T_intentPred | T_intentGT

T_intentEval: TypeAlias = dict[
    {
        "acc": float,
        "f1": float,
        "mAcc": float,
        "confusion_matrix": npt.NDArray[np.int_ | np.float_],
    }
]
"""Intent evaluation type alias.

A dictionary with the following structure:
{
    "acc": float,
    "f1": float,
    "mAcc": float,
    "confusion_matrix": np.ndarray
}
"""

T_trajPredGT: TypeAlias = dict[
    str, dict[str, dict[str, dict[{"traj": list[list[float]]}]]]
]
"""Trajectory ground truth and prediction type alias.

A dictionary with the following structure:
{
    "Video ID": {
        "Pedestrian ID": {
            "Frame ID": {
                "traj": list[list[float]]
            }
        }
    }
}
"""

T_trajSubEval: TypeAlias = dict[{"0.5": float, "1.0": float, "1.5": float}]
"""Trajectory sub-evaluation type alias.

A dictionary with the following structure:
{
    "0.5": float,
    "1.0": float,
    "1.5": float
}
"""

T_trajEval: TypeAlias = dict[
    {
        "ADE": T_trajSubEval,
        "FDE": T_trajSubEval,
        "ARB": T_trajSubEval,
        "FRB": T_trajSubEval,
    }
]
"""Trajectory evaluation type alias.

A dictionary with the following structure:
{
    "ADE": T_trajSubEval,
    "FDE": T_trajSubEval,
    "ARB": T_trajSubEval,
    "FRB": T_trajSubEval
}
"""

T_drivingPredGT: TypeAlias = dict[
    str, dict[str, dict[{"speed": int, "direction": int}]]
]
"""Driving ground truth and prediction type alias.

A dictionary with the following structure:
{
    "Video ID": {
        "Frame ID": {
            "speed": int,
            "direction": int
        }
    }
}
"""

T_drivingEval: TypeAlias = dict[
    {
        "speed_Acc": float,
        "speed_mAcc": float,
        "speed_confusion_matrix": npt.NDArray[np.int_ | np.float_],
        "direction_Acc": float,
        "direction_mAcc": float,
        "dir_confusion_matrix": npt.NDArray[np.int_ | np.float_],
    }
]
"""Driving evaluation type alias.

A dictionary with the following structure:
{
    "speed_Acc": float,
    "speed_mAcc": float,
    "speed_confusion_matrix": np.ndarray,
    "direction_Acc": float,
    "direction_mAcc": float,
    "dir_confusion_matrix": np.ndarray
}
"""


def evaluate_intent(
    groundtruth: str | os.PathLike[Any],
    prediction: str | os.PathLike[Any],
    args: DefaultArguments,
) -> tuple[float, float, float]:
    """Evaluates intent from ground truth and prediction json files.

    :param groundtruth: Path to ground truth json, defaults to "".
    :type groundtruth: str | os.PathLike
    :param prediction: Path to prediction json, defaults to "".
    :type prediction: str | os.PathLike
    :param args: Arguments for running training functions.
    :type args: DefaultArgument

    :return: Tuple of accuracy, f1 score, and mAcc.
    :rtype: tuple[float, float, float]
    """
    with open(groundtruth, "r") as f:
        gt_intent: T_intentGT = json.load(f)

    with open(prediction, "r") as f:
        pred_intent: T_intentPred = json.load(f)

    gt: list[int] = []
    pred: list[int] = []
    for vid in gt_intent.keys():
        for pid in gt_intent[vid].keys():
            for fid in gt_intent[vid][pid].keys():
                gt.append(gt_intent[vid][pid][fid]["intent"])
                pred.append(pred_intent[vid][pid][fid]["intent"])

    gt_np: npt.NDArray[np.int_] = np.array(gt)
    pred_np: npt.NDArray[np.int_] = np.array(pred)
    res = measure_intent_prediction(gt_np, pred_np, args)
    print("Acc: ", res["acc"])
    print("F1: ", res["f1"])
    print("mAcc: ", res["mAcc"])
    print("ConfusionMatrix: ", res["confusion_matrix"])
    return res["acc"], res["f1"], res["mAcc"]


def measure_intent_prediction(
    target: npt.NDArray[np.int_],
    prediction: npt.NDArray[np.int_],
    args: DefaultArguments,
) -> T_intentEval:
    """Evaluates intent from ground truth and prediction json files loaded from
         :py:func:`evaluate_intent`.

    :param npt.NDArray[np.int_] target: Ground truth targets.
     :param npt.NDArray[np.int_] prediction: Model predictions.
     :param DefaultArguments args: Training arguments.

     :returns: Accuracy, F1 Score, mAccuracy, and Confusion Matrix in a dictionary.
     :rtype: T_intentEval
    """
    print("Evaluating Intent ...")
    lbl_target = target  # bs
    lbl_pred = np.round(prediction)  # bs, use 0.5 as threshold

    results: T_intentEval = {
        "acc": 0.0,
        "f1": 0.0,
        "mAcc": 0.0,
        "confusion_matrix": None,  # type: ignore[reportAssignmentType]
    }

    # hard label evaluation - acc, f1
    acc: float = accuracy_score(lbl_target, lbl_pred)  # calculate acc for all samples
    f1: float = f1_score(lbl_target, lbl_pred, average="macro")

    intent_matrix = confusion_matrix(lbl_target, lbl_pred)  # [2 x 2]
    intent_cls_acc = np.array(
        intent_matrix.diagonal() / intent_matrix.sum(axis=-1)
    )  # 2
    intent_cls_mean_acc: float = intent_cls_acc.mean(axis=0)

    results["acc"] = acc
    results["f1"] = f1
    results["mAcc"] = intent_cls_mean_acc
    results["confusion_matrix"] = intent_matrix

    return results


def evaluate_traj(
    groundtruth: str | os.PathLike[Any],
    prediction: str | os.PathLike[Any],
    args: DefaultArguments,
) -> np.float_:
    """Evaluates trajectory from ground truth and prediction json files.

    :param groundtruth: Path to ground truth json, defaults to "".
    :type groundtruth: str | os.PathLike
    :param prediction: Path to prediction json, defaults to "".
    :type prediction: str | os.PathLike
    :param args: Arguments for running training functions.
    :type args: DefaultArgument

    :return: Mean of average displacement error.
    :rtype: np.float_
    """
    with open(groundtruth, "r") as f:
        gt_traj: T_trajPredGT = json.load(f)

    with open(prediction, "r") as f:
        pred_traj: T_trajPredGT = json.load(f)

    gt: list[list[list[float]]] = []
    pred: list[list[list[float]]] = []
    for vid in gt_traj.keys():
        for pid in gt_traj[vid].keys():
            for fid in gt_traj[vid][pid].keys():
                gt.append(gt_traj[vid][pid][fid]["traj"])
                pred.append(pred_traj[vid][pid][fid]["traj"])
    gt_np: npt.NDArray[np.float_] = np.array(gt)
    pred_np: npt.NDArray[np.float_] = np.array(pred)
    traj_results = measure_traj_prediction(gt_np, pred_np, args)

    for key in [
        "ADE",
        "FDE",
        "ARB",
        "FRB",
    ]:  # , 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
        for time in ["0.5", "1.0", "1.5"]:
            val: float = traj_results[key][time]
            print(f"Eval/Results/{key}_{time}", val)

    score = np.mean([traj_results["ADE"][t] for t in ["0.5", "1.0", "1.5"]])
    return score


def measure_traj_prediction(
    target: npt.NDArray[np.float_],
    prediction: npt.NDArray[np.float_],
    args: DefaultArguments,
) -> T_trajEval:
    """Evaluates intent from ground truth and prediction json files loaded from
        :py:func:`evaluate_traj`.

    :param npt.NDArray[np.int_] target: Ground truth targets.
    :param npt.NDArray[np.int_] prediction: Model predictions.
    :param DefaultArguments args: Training arguments.

    :returns: Dictionary of metrics measured at [0.5s, 1.0s, 1.5s] in the future.
    .. hlist::
        * `ADE`: Average Displacement Error (Based on the centre of the bounding box)
        * `FDE`: Final Displacement Error (Based on the centre of the bounding box)
        * `ARB`: Average RMSE of Bounding Box coordinates
        * `FRB`: Final RMSE of Bounding Box coordinates
    :rtype: T_trajEval
    """
    print("Evaluating Trajectory ...")
    target = np.array(target)
    prediction = np.array(prediction)
    assert target.shape[1] == args.predict_length
    assert target.shape[2] == 4  # bbox
    assert prediction.shape[1] == args.predict_length
    assert prediction.shape[2] == 4
    results: T_trajEval = {
        # 'Bbox_MSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        # 'Bbox_FMSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        # 'Center_MSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        # 'Center_FMSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        "ADE": {"0.5": 0, "1.0": 0, "1.5": 0},  # center
        "FDE": {"0.5": 0, "1.0": 0, "1.5": 0},  # center
        "ARB": {"0.5": 0, "1.0": 0, "1.5": 0},  # bbox - B: bbox
        "FRB": {"0.5": 0, "1.0": 0, "1.5": 0},  # bbox - B: bbox
    }
    bs, ts, _ = target.shape
    # Error: performance_MSE = np.square(target - prediction).sum(axis=2)  # n_samples x ts x 4 --> bs x ts
    performance_MSE: npt.NDArray[np.float_] = np.square(target - prediction).mean(
        axis=2
    )
    performance_RMSE: npt.NDArray[np.float_] = np.sqrt(performance_MSE)  # bs x ts
    for t in [0.5, 1.0, 1.5]:
        end_frame = int(t * args.fps)
        #     # 1. bbox MSE
        #     results['Bbox_MSE'][str(t)] = performance_MSE[:, :end_frame].mean(axis=None)
        #     # 2. bbox FMSE
        #     results['Bbox_FMSE'][str(t)] = performance_MSE[:, end_frame - 1].mean(axis=None)
        #
        # 5. ARB - bbox
        results["ARB"][str(t)] = performance_RMSE[:, :end_frame].mean(axis=None)
        # 6. FRB - bbox
        results["FRB"][str(t)] = performance_RMSE[:, end_frame - 1].mean(axis=None)

    # centers
    center_target = np.zeros((bs, ts, 2))
    center_pred = np.zeros((bs, ts, 2))
    for i in range(bs):
        for j in range(ts):
            center_target[i, j, 0] = (target[i, j, 0] + target[i, j, 2]) / 2
            center_target[i, j, 1] = (target[i, j, 1] + target[i, j, 3]) / 2
            center_pred[i, j, 0] = (prediction[i, j, 0] + prediction[i, j, 2]) / 2
            center_pred[i, j, 1] = (prediction[i, j, 1] + prediction[i, j, 3]) / 2

    performance_CMSE: npt.NDArray[np.float_] = np.square(
        center_target - center_pred
    ).sum(
        axis=2
    )  # bs x ts x 4 --> bs x ts
    performance_CRMSE: npt.NDArray[np.float_] = np.sqrt(performance_CMSE)  # bs x ts

    for t in [0.5, 1.0, 1.5]:
        end_frame = int(t * args.fps)
        # # 3. C_MSE
        # results['Center_MSE'][str(t)] = performance_CMSE[:, :end_frame].mean(axis=None)
        # # 4. C_FMSE
        # results['Center_FMSE'][str(t)] = performance_CMSE[:, end_frame - 1].mean(axis=None)
        # 7. ADE - center
        results["ADE"][str(t)] = performance_CRMSE[:, :end_frame].mean(axis=None)
        # 8. FDE - center
        results["FDE"][str(t)] = performance_CRMSE[:, end_frame - 1].mean(axis=None)

    return results


def evaluate_driving(
    groundtruth: str | os.PathLike[Any],
    prediction: str | os.PathLike[Any],
    args: DefaultArguments,
) -> float:
    """Evaluate driving performance from dumped json files.

    :param groundtruth: Path to ground truth json, defaults to "".
    :type groundtruth: str | os.PathLike
    :param prediction: Path to prediction json, defaults to "".
    :type prediction: str | os.PathLike
    :param args: Arguments for running the program.
    :type args: DefaultArguments

    :return: Mean of `speed_mAcc` and `direction_mAcc`.
    :rtype: float
    """
    with open(groundtruth, "r") as f:
        gt_driving: T_drivingPredGT = json.load(f)

    with open(prediction, "r") as f:
        pred_driving: T_drivingPredGT = json.load(f)

    gt_speed: list[int] = []
    gt_dir: list[int] = []
    speed_pred: list[int] = []
    dir_pred: list[int] = []

    for vid in pred_driving.keys():
        for fid in pred_driving[vid].keys():
            speed_pred.append(pred_driving[vid][fid]["speed"])
            dir_pred.append(pred_driving[vid][fid]["direction"])
            gt_speed.append(gt_driving[vid][fid]["speed"])
            gt_dir.append(gt_driving[vid][fid]["direction"])

    gt_speed_np: npt.NDArray[np.int_] = np.array(gt_speed)
    gt_dir_np: npt.NDArray[np.int_] = np.array(gt_dir)
    speed_pred_np: npt.NDArray[np.int_] = np.array(speed_pred)
    dir_pred_np: npt.NDArray[np.int_] = np.array(dir_pred)

    res = measure_driving_prediction(
        gt_speed_np, gt_dir_np, speed_pred_np, dir_pred_np, args
    )
    for key in [
        "speed_Acc",
        "speed_mAcc",
        "direction_Acc",
        "direction_mAcc",
        "speed_confusion_matrix",
        "dir_confusion_matrix",
    ]:
        print(key, res[key])

    score = (res["speed_mAcc"] + res["direction_mAcc"]) / 2
    return score


def measure_driving_prediction(
    gt_speed: npt.NDArray[np.int_],
    gt_dir: npt.NDArray[np.int_],
    speed_pred: npt.NDArray[np.int_],
    dir_pred: npt.NDArray[np.int_],
    args: DefaultArguments,
) -> T_drivingEval:
    """Evaluates driving predictions from ground truth and prediction json files
        loaded from :py:func:`evaluate_driving`.

    :param npt.NDArray[np.int_] gt_speed: Ground truth speed decision targets.
    :param npt.NDArray[np.int_] gt_dir: Ground truth direction decision targets.
    :param npt.NDArray[np.int_] speed_pred: Speed decision predictions.
    :param npt.NDArray[np.int_] dir_pred: Direction decision predictions.
    :param DefaultArguments args: Training arguments.

    :returns: Dictionary containing acc, mAcc, and confusion matrices for speed and
        direction predictions.
    :rtype: T_drivingEval
    """
    results: T_drivingEval = {  # type: ignore[reportAssignmentType]
        "speed_Acc": 0.0,
        "speed_mAcc": 0.0,
        "speed_confusion_matrix": None,
        "direction_Acc": 0.0,
        "direction_mAcc": 0.0,
        "dir_confusion_matrix": None,
    }
    print("Evaluating Driving Decision Prediction ...")

    results["speed_Acc"] = accuracy_score(gt_speed, speed_pred)
    results["direction_Acc"] = accuracy_score(gt_dir, dir_pred)

    speed_matrix = confusion_matrix(gt_speed, speed_pred)
    print("Speed matrix: ", speed_matrix)
    results["speed_confusion_matrix"] = speed_matrix
    sum_cnt = speed_matrix.sum(axis=1)
    sum_cnt = np.array([max(1, i) for i in sum_cnt])
    speed_cls_wise_acc = speed_matrix.diagonal() / sum_cnt
    results["speed_mAcc"] = np.mean(speed_cls_wise_acc)

    dir_matrix = confusion_matrix(gt_dir, dir_pred)
    print("Dir matrix: ", dir_matrix)
    results["dir_confusion_matrix"] = dir_matrix
    sum_cnt = dir_matrix.sum(axis=1)
    sum_cnt = np.array([max(1, i) for i in sum_cnt])
    dir_cls_wise_acc = dir_matrix.diagonal() / sum_cnt
    results["direction_mAcc"] = np.mean(dir_cls_wise_acc)

    # print("dir: ", dir_matrix.diagonal(), sum_cnt, dir_cls_wise_acc, np.mean(dir_cls_wise_acc))
    return results


if __name__ == "__main__":
    args = None
    # evaluate_intent('gt.json', 'pred.json', args)
    test_gt_file = "../val_intent_gt.json"
    test_pred_file = "../val_intent_pred"
    score, _, _ = evaluate_intent(test_gt_file, test_pred_file, args)
    print("Ranking score is : ", score)
