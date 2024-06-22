import json
import os
from typing import Any, Optional
from dataclasses import dataclass

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
        gt_intent = json.load(f)

    with open(prediction, "r") as f:
        pred_intent = json.load(f)

    gt = []
    pred = []
    for vid in gt_intent.keys():
        for pid in gt_intent[vid].keys():
            for fid in gt_intent[vid][pid].keys():
                gt.append(gt_intent[vid][pid][fid]["intent"])
                pred.append(pred_intent[vid][pid][fid]["intent"])
    gt = np.array(gt)
    pred = np.array(pred)
    res = measure_intent_prediction(gt, pred, args)
    print("Acc: ", res.acc)
    print("F1: ", res.f1)
    print("mAcc: ", res.mAcc)
    print("ConfusionMatrix: ", res.confusion_matrix)
    return res.acc, res.f1, res.mAcc


@dataclass
class IntentPrediction:
    acc: float
    f1: float
    mAcc: float
    confusion_matrix: npt.NDArray[np.int32 | np.int64 | np.float32 | np.float64]


def measure_intent_prediction(
    target: npt.NDArray[np.float32 | np.float64],
    prediction: npt.NDArray[np.float32 | np.float64],
    args: DefaultArguments,
) -> IntentPrediction:
    print("Evaluating Intent ...")
    results = {
        "Acc": 0,
        "F1": 0,
        "mAcc": 0,
        "ConfusionMatrix": [[]],
    }

    bs = target.shape[0]
    lbl_target = target  # bs
    lbl_pred = np.round(prediction)  # bs, use 0.5 as threshold

    # hard label evaluation - acc, f1
    Acc: float = accuracy_score(lbl_target, lbl_pred)  # calculate acc for all samples
    F1_score = f1_score(lbl_target, lbl_pred, average="macro")

    intent_matrix = confusion_matrix(lbl_target, lbl_pred)  # [2 x 2]
    intent_cls_acc = np.array(
        intent_matrix.diagonal() / intent_matrix.sum(axis=-1)
    )  # 2
    intent_cls_mean_acc = intent_cls_acc.mean(axis=0)

    # results['MSE'] = MSE
    # results["Acc"] = Acc
    # results["F1"] = F1_score
    # results["mAcc"] = intent_cls_mean_acc
    # results["ConfusionMatrix"] = intent_matrix
    return IntentPrediction(Acc, F1_score, intent_cls_mean_acc, intent_matrix)


def evaluate_traj(
    groundtruth: str | os.PathLike[Any],
    prediction: str | os.PathLike[Any],
    args: DefaultArguments,
) -> np.floating[Any]:
    with open(groundtruth, "r") as f:
        gt_traj = json.load(f)

    with open(prediction, "r") as f:
        pred_traj = json.load(f)

    gt = []
    pred = []
    for vid in gt_traj.keys():
        for pid in gt_traj[vid].keys():
            for fid in gt_traj[vid][pid].keys():
                gt.append(gt_traj[vid][pid][fid]["traj"])
                pred.append(pred_traj[vid][pid][fid]["traj"])
    gt_np: npt.NDArray[np.float32 | np.float64] = np.array(gt)
    pred_np: npt.NDArray[np.float32 | np.float64] = np.array(pred)
    traj_results = measure_traj_prediction(gt_np, pred_np, args)

    for key in [
        "ADE",
        "FDE",
        "ARB",
        "FRB",
    ]:  # , 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
        for time in ["0.5", "1.0", "1.5"]:
            val = traj_results[key][time]
            print(f"Eval/Results/{key}_{time}", val)

    score = np.mean([traj_results["ADE"][t] for t in ["0.5", "1.0", "1.5"]])
    return score


def measure_traj_prediction(
    target: npt.NDArray[np.float32 | np.float64],
    prediction: npt.NDArray[np.float32 | np.float64],
    args: DefaultArguments,
) -> dict[str, dict[str, int]]:
    print("Evaluating Trajectory ...")
    target = np.array(target)
    prediction = np.array(prediction)
    assert target.shape[1] == args.predict_length
    assert target.shape[2] == 4  # bbox
    assert prediction.shape[1] == args.predict_length
    assert prediction.shape[2] == 4
    results = {
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
    performance_MSE = np.square(target - prediction).mean(axis=2)
    performance_RMSE = np.sqrt(performance_MSE)  # bs x ts
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

    performance_CMSE = np.square(center_target - center_pred).sum(
        axis=2
    )  # bs x ts x 4 --> bs x ts
    performance_CRMSE = np.sqrt(performance_CMSE)  # bs x ts

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
        gt_driving = json.load(f)

    with open(prediction, "r") as f:
        pred_driving = json.load(f)

    gt_speed = []
    gt_dir = []
    speed_pred = []
    dir_pred = []

    for vid in pred_driving.keys():
        for fid in pred_driving[vid].keys():
            speed_pred.append(pred_driving[vid][fid]["speed"])
            dir_pred.append(pred_driving[vid][fid]["direction"])
            gt_speed.append(gt_driving[vid][fid]["speed"])
            gt_dir.append(gt_driving[vid][fid]["direction"])

    gt_speed = np.array(gt_speed)
    gt_dir = np.array(gt_dir)
    speed_pred = np.array(speed_pred)
    dir_pred = np.array(dir_pred)

    res = measure_driving_prediction(gt_speed, gt_dir, speed_pred, dir_pred, args)
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
    gt_speed, gt_dir, speed_pred, dir_pred, args
) -> dict[str, Any]:
    results = {
        "speed_Acc": 0.0,
        "speed_mAcc": 0.0,
        "speed_confusion_matrix": None,
        "direction_Acc": 0.0,
        "direction_mAcc": 0.0,
        "dir_confusion_matrix": None,
    }
    print("Evaluating Driving Decision Prediction ...")

    bs = gt_speed.shape[0]
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
