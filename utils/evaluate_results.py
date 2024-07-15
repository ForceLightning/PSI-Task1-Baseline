"""Evaluation functions for intent, trajectory, and driving prediction.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import typing as npt
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from tqdm.auto import tqdm

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
    str,
    dict[
        str,
        dict[
            {
                "speed": int,
                "speed_proba": list[float],
                "direction": int,
                "direction_proba": list[float],
            }
        ],
    ],
]
"""Driving ground truth and prediction type alias.

A dictionary with the following structure:
{
    "Video ID": {
        "Frame ID": {
            "speed": int,
            "speed_proba": list[float],
            "direction": int
            "direction_proba": list[float]
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
    gen_auc_charts: bool = False,
) -> tuple[float, float, float, npt.NDArray[np.int_ | np.float_]]:
    """Evaluates intent from ground truth and prediction json files.

    :param groundtruth: Path to ground truth json, defaults to "".
    :type groundtruth: str | os.PathLike
    :param prediction: Path to prediction json, defaults to "".
    :type prediction: str | os.PathLike
    :param args: Arguments for running training functions.
    :type args: DefaultArgument
    :param bool gen_auc_charts: Whether to generate AUROC and AUPRC charts.

    :return: Tuple of accuracy, f1 score, mAcc, and confusion matrix.
    :rtype: tuple[float, float, float, npt.NDArray[np.int_ | np.float_]]
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

    if gen_auc_charts:
        fig = auc_charts([gt_np], [pred_np], [args.model_name], args.task_name)
        fig.savefig(
            os.path.join(args.checkpoint_path, "results", "charts.png"),
            transparent=True,
        )

    print("Acc: ", res["acc"])
    print("F1: ", res["f1"])
    print("mAcc: ", res["mAcc"])
    print("ConfusionMatrix: ", res["confusion_matrix"])
    return res["acc"], res["f1"], res["mAcc"], res["confusion_matrix"]


def evaluate_intents(
    paths: list[tuple[str, str]],
    gen_auc_charts: bool = False,
    auc_chart_labels: list[str] | None = None,
) -> (
    tuple[list[tuple[float, float, float, npt.NDArray[np.int_ | np.float_]]], Figure]
    | list[tuple[float, float, float, npt.NDArray[np.int_ | np.float_]]]
):
    """Evaluates intent from ground truth and prediction json files.

    :param paths: Tuple of paths to ground truth and prediction json files, defaults to
    "".
    :param DefaultArgument args: Arguments for running training functions.
    :param bool gen_auc_charts: Whether to generate AUROC and AUPRC charts, defaults to
    False.
    :param auc_chart_labels: List of model names, defaults to None.
    :type auc_chart_labels: list[str] or None

    :return: List of tuples of accuracy, f1 score, mAcc, and confusion matrix, one tuple
    for each model, or the aforementioned list and a figure as a tuple if
    `gen_auc_charts` is True.
    :rtype: tuple[list[tuple[float, float, float, NDArray[int_ | float_]]], Figure] or
    list[tuple[float, float, float, NDArray[int_ | float_]]]
    """
    gts: list[npt.NDArray[np.int_]] = []
    preds: list[npt.NDArray[np.int_]] = []
    probas: list[npt.NDArray[np.float_]] = []

    results: list[tuple[float, float, float, npt.NDArray[np.int_ | np.float_]]] = []

    if auc_chart_labels is None:
        auc_chart_labels = [f"model_{i}" for i in range(len(paths))]

    paths_iter = tqdm(zip(paths, auc_chart_labels), desc="Evaluating Intent")
    for (gt_path, pred_path), model_name in paths_iter:
        paths_iter.set_postfix_str(f"Model: {model_name}")

        gt: list[int] = []
        pred: list[int] = []
        proba: list[float] = []

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_intent: T_intentGT = json.load(f)

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_intent: T_intentPred = json.load(f)

        for vid in gt_intent:
            for pid in gt_intent[vid]:
                for fid in gt_intent[vid][pid]:
                    gt.append(gt_intent[vid][pid][fid]["intent"])
                    pred.append(pred_intent[vid][pid][fid]["intent"])
                    proba.append(pred_intent[vid][pid][fid]["intent_prob"])

        np_gt = np.array(gt)
        np_pred = np.array(pred)

        # GUARD
        assert (
            np_gt.shape == np_pred.shape
        ), f"Ground truth and prediction shapes {np_gt.shape} != {np_pred.shape} mismatch"

        bal_acc = balanced_accuracy_score(np_gt, np_pred)

        classi_report = classification_report(
            np_gt, np_pred, target_names=["not cross", "cross"], output_dict=True
        )

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.expand_frame_repr",
            False,
        ):
            paths_iter.write(f"Model: {model_name}")
            paths_iter.write(str(pd.DataFrame(classi_report)))
            paths_iter.write(f"Balanced Accuracy: {bal_acc:.4f}")

        res = measure_intent_prediction(np_gt, np_pred)

        gts.append(np_gt)
        preds.append(np_pred)
        probas.append(np.array(proba))
        results.append((res["acc"], res["f1"], res["mAcc"], res["confusion_matrix"]))

    if gen_auc_charts:
        fig = auc_charts(gts, probas, auc_chart_labels, "ped_intent", None, True)
        return results, fig
    return results


def auc_charts(
    gts: list[npt.NDArray[np.int_]],
    probas: list[npt.NDArray[np.float_]],
    labels: list[str] | None,
    task_name: str = "ped_intent",
    average: Literal["macro", "weighted", "micro"] | None = None,
    horizontal: bool = False,
) -> Figure:
    """Generates ROC and PRC charts as a comparison amongst various model results.

    :param gts: List of ground truth labels, 1 ndarray for each model.
    :type gts: list[npt.NDArray[np.int_]]
    :param probas: List of predicted probabilities of labels, 1 ndarray for each model.
    :type probas: list[npt.NDArray[np.float_]]
    :param list[str] labels: Names of the models.
    :param task_name: Name of the task, defaults to "ped_intent".
    :type task_name: str
    :param bool horizontal: Whether to display the charts horizontally, defaults to False.

    :return: Figure with ROC and PRC charts.
    :rtype: Figure
    """
    sns.set_theme("paper", "whitegrid")
    if horizontal:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11.69, 5.15), dpi=100)
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8.27, 11.69), dpi=100)

    labels = [f"model_{i}" for i in range(len(gts))] if labels is None else labels

    for gt, proba, label in zip(gts, probas, labels):
        # ascertain target type
        y_type = type_of_target(gt)
        if y_type == "binary":
            _binary_auc_charts(fig, ax, gt, proba, label)
        elif y_type == "multiclass":
            average = average if average else "macro"
            _multiclass_auc_charts(fig, ax, gt, proba, label, average)
        else:
            raise ValueError(f"Target type ({y_type}) not supported")

    title = (
        f"ROC and PRC Curves for {task_name}"
        if task_name is not None
        else "ROC and PRC Curves"
    )
    fig.suptitle(title)

    return fig


def _binary_auc_charts(
    fig: Figure,
    ax: Sequence[Axes],
    gt: npt.NDArray[np.int_],
    proba: npt.NDArray[np.float_],
    label: str,
) -> None:
    fpr, tpr, _ = sk_roc_curve(gt, proba)
    auroc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(gt, proba)
    auprc = average_precision_score(gt, proba)

    _ = ax[0].plot(fpr, tpr, label=f"{label} AUROC: {auroc:.4f}", lw=2, alpha=0.8)
    _ = ax[0].set_xlabel("False Positive Rate")
    _ = ax[0].set_ylabel("True Positive Rate")
    _ = ax[0].set_title("Receiver Operating Characteristics")
    _ = ax[0].set_ylim(-0.1, 1.1)
    _ = ax[0].legend(loc="lower right")

    _ = ax[1].plot(rec, prec, label=f"{label} AUPRC: {auprc:.4f}", lw=2, alpha=0.8)
    _ = ax[1].set_xlabel("Recall")
    _ = ax[1].set_ylabel("Precision")
    _ = ax[1].set_title("Precision Recall Curve")
    _ = ax[1].set_ylim(-0.1, 1.1)
    _ = ax[1].legend(loc="lower right")


def _multiclass_auc_charts(
    fig: Figure,
    ax: Sequence[Axes],
    gt: npt.NDArray[np.int_],
    proba: npt.NDArray[np.float_],
    label: str,
    average: Literal["macro", "weighted", "micro"] = "macro",
) -> None:
    roc_label_binarizer = LabelBinarizer().fit(gt)
    gt = roc_label_binarizer.transform(gt)
    num_classes = roc_label_binarizer.classes_.shape[0]

    intermediate_tprs, intermediate_tpr_threshes = [], []
    intermediate_precs, intermediate_recall_threshes = [], []

    recall_mean = np.linspace(0, 1, 1000)
    fpr_mean = np.linspace(0, 1, 1000)

    for i in range(num_classes):
        if np.sum(gt, axis=0)[i] == 0:
            continue

        fpr, tpr, tpr_thresh = sk_roc_curve(gt[:, i], proba[:, i])
        intermediate_tpr_threshes.append(tpr_thresh[np.argmin(np.abs(tpr - fpr))])
        tpr_interp = np.interp(fpr_mean, fpr, tpr)
        tpr_interp[0] = 0.0
        intermediate_tprs.append(tpr_interp)

        prec, rec, recall_thresh = precision_recall_curve(gt[:, i], proba[:, i])
        prec_interp = np.interp(recall_mean, rec[::-1], prec[::-1])
        intermediate_precs.append(prec_interp)
        intermediate_recall_threshes.append(
            recall_thresh[np.argmin(np.abs(prec - rec))]
        )

    auroc = roc_auc_score(gt, proba, average=average)
    auprc = average_precision_score(gt, proba, average=average)

    match average:
        case "macro":
            tpr = np.mean(intermediate_tprs, axis=0)
            tpr_thresh = np.mean(intermediate_tpr_threshes)
            prec = np.mean(intermediate_precs, axis=0)
            recall_thresh = np.mean(intermediate_recall_threshes)
        case "weighted":
            class_distribution = np.sum(gt, axis=0)
            if len(class_distribution) < num_classes:
                class_distribution = np.append(
                    class_distribution, [0] * (num_classes - len(class_distribution))
                )

            class_distribution = class_distribution / np.sum(class_distribution)
            tpr = np.array(intermediate_tprs).T @ class_distribution
            prec = np.array(intermediate_precs).T @ class_distribution
        case _:
            raise NotImplementedError(f"Average type {average} not supported")

    _ = ax[0].plot(fpr_mean, tpr, label=f"{label} AUROC: {auroc:.4f}", lw=2, alpha=0.8)
    _ = ax[0].set_xlabel("False Positive Rate")
    _ = ax[0].set_ylabel("True Positive Rate")
    _ = ax[0].set_title("Receiver Operating Characteristics")
    _ = ax[0].set_ylim(-0.1, 1.1)
    _ = ax[0].legend(loc="lower right")

    _ = ax[1].plot(
        recall_mean, prec, label=f"{label} AUPRC: {auprc:.4f}", lw=2, alpha=0.8
    )
    _ = ax[1].set_xlabel("Recall")
    _ = ax[1].set_ylabel("Precision")
    _ = ax[1].set_title("Precision Recall Curve")
    _ = ax[1].set_ylim(-0.1, 1.1)
    _ = ax[1].legend(loc="lower right")


def measure_intent_prediction(
    target: npt.NDArray[np.int_],
    prediction: npt.NDArray[np.int_],
) -> T_intentEval:
    """Evaluates intent from ground truth and prediction json files loaded from
         :py:func:`evaluate_intent`.

    :param npt.NDArray[np.int_] target: Ground truth targets.
     :param npt.NDArray[np.int_] prediction: Model predictions.

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


def evaluate_trajs(paths: list[tuple[str, str]]) -> pd.DataFrame:
    results: dict[str, list[float] | list[str]] = {
        "model_name": [],
        "ADE_0.5": [],
        "ADE_1.0": [],
        "ADE_1.5": [],
        "FDE_0.5": [],
        "FDE_1.0": [],
        "FDE_1.5": [],
        "ARB_0.5": [],
        "ARB_1.0": [],
        "ARB_1.5": [],
        "FRB_0.5": [],
        "FRB_1.0": [],
        "FRB_1.5": [],
    }

    paths_iter = tqdm(paths, desc="Evaluating Trajectory")
    for gt_path, pred_path in paths_iter:
        paths_iter.set_postfix_str("Model: " + os.path.basename(gt_path))
        gt: list[npt.NDArray[np.float_]] = []
        pred: list[npt.NDArray[np.float_]] = []

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_traj: T_trajPredGT = json.load(f)

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_traj: T_trajPredGT = json.load(f)

        for vid in gt_traj:
            for pid in gt_traj[vid]:
                for fid in gt_traj[vid][pid]:
                    try:
                        gt_sample = gt_traj[vid][pid][fid]["traj"]
                        pred_sample = pred_traj[vid][pid][fid]["traj"]

                        np_gt_sample = np.array(gt_sample)
                        np_pred_sample = np.array(pred_sample)

                        if np_pred_sample.max() > 1 and np_gt_sample.max() <= 1:
                            np_pred_sample[:, 0] /= 1280
                            np_pred_sample[:, 2] /= 1280
                            np_pred_sample[:, 1] /= 720
                            np_pred_sample[:, 3] /= 720

                        gt.append(np_gt_sample)
                        pred.append(np_pred_sample)
                    except KeyError:
                        continue

        args = DefaultArguments()
        args.predict_length = 45
        args.observe_length = 15
        args.max_track_size = 60

        gt_np: npt.NDArray[np.float_] = np.array(gt)
        pred_np: npt.NDArray[np.float_] = np.array(pred)
        traj_results = measure_traj_prediction(gt_np, pred_np, args)

        model_name = os.path.normpath(pred_path).split(os.sep)[-4]

        results["model_name"].append(model_name)

        for key in ["ADE", "FDE", "ARB", "FRB"]:
            for time in ["0.5", "1.0", "1.5"]:
                val: float = traj_results[key][time]
                results[f"{key}_{time}"].append(val)

    df = pd.DataFrame(results)
    return df


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


def evaluate_drivings(
    paths: list[tuple[str, str]],
    gen_auc_charts: bool = False,
    auc_chart_labels: list[str] | None = None,
):
    gts_speeds: list[npt.NDArray[np.int_]] = []
    gts_dirs: list[npt.NDArray[np.int_]] = []
    preds_speeds: list[npt.NDArray[np.int_]] = []
    preds_dir: list[npt.NDArray[np.int_]] = []
    probas_speeds: list[npt.NDArray[np.float_]] = []
    probas_dir: list[npt.NDArray[np.float_]] = []

    results: list[T_drivingEval] = []

    if auc_chart_labels is None:
        auc_chart_labels = [f"model_{i}" for i in range(len(paths))]

    path_iter = tqdm(zip(paths, auc_chart_labels))
    for (gt_path, pred_path), model_name in path_iter:
        path_iter.set_postfix_str("Model: " + model_name)

        gt_speed: list[int] = []
        gt_dir: list[int] = []
        pred_speed: list[int] = []
        pred_dir: list[int] = []
        proba_speed: list[list[float]] = []
        proba_dir: list[list[float]] = []

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_driving: T_drivingPredGT = json.load(f)

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_driving: T_drivingPredGT = json.load(f)

        for vid in pred_driving:
            for fid in pred_driving[vid]:
                pred_speed.append(pred_driving[vid][fid]["speed"])
                pred_dir.append(pred_driving[vid][fid]["direction"])
                proba_speed.append(pred_driving[vid][fid]["speed_proba"])
                proba_dir.append(pred_driving[vid][fid]["direction_proba"])
                gt_speed.append(gt_driving[vid][fid]["speed"])
                gt_dir.append(gt_driving[vid][fid]["direction"])

        np_gt_speed = np.array(gt_speed)
        np_gt_dir = np.array(gt_dir)
        np_pred_speed = np.array(pred_speed)
        np_pred_dir = np.array(pred_dir)
        np_proba_speed = np.array(proba_speed)
        np_proba_dir = np.array(proba_dir)

        res = measure_driving_prediction(
            np_gt_speed,
            np_gt_dir,
            np_pred_speed,
            np_pred_dir,
            DefaultArguments(),
            verbose=False,
        )
        speed_bal_acc = balanced_accuracy_score(
            np_gt_speed,
            np_pred_speed,
            # adjusted=True
        )
        dir_bal_acc = balanced_accuracy_score(
            np_gt_dir,
            np_pred_dir,
            # adjusted=True
        )

        speed_classi_report = classification_report(
            np_gt_speed,
            np_pred_speed,
            target_names=["maintainSpeed", "decreaseSpeed", "increaseSpeed"],
            output_dict=True,
        )

        dir_classi_report = classification_report(
            np_gt_dir,
            np_pred_dir,
            target_names=["goStraight", "turnLeft", "turnRight"],
            output_dict=True,
        )

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.expand_frame_repr",
            False,
        ):
            path_iter.write(f"Model: {model_name}")
            path_iter.write("Speed classification report: ")
            path_iter.write(str(pd.DataFrame(speed_classi_report)))
            path_iter.write(f"Speed Balanced Accuracy: {speed_bal_acc:.4f}")
            path_iter.write("Speed confusion matrix: ")
            path_iter.write(str(pd.DataFrame(res["speed_confusion_matrix"])))

            path_iter.write("Direction classification report: ")
            path_iter.write(str(pd.DataFrame(dir_classi_report)))
            path_iter.write(f"Direction Balanced Accuracy: {dir_bal_acc:.4f}")
            path_iter.write("Direction confusion matrix: ")
            path_iter.write(str(pd.DataFrame(res["dir_confusion_matrix"])))

        results.append(res)

        gts_speeds.append(np_gt_speed)
        gts_dirs.append(np_gt_dir)
        preds_speeds.append(np_pred_speed)
        preds_dir.append(np_pred_dir)
        probas_speeds.append(np_proba_speed)
        probas_dir.append(np_proba_dir)

    if gen_auc_charts:
        fig_speed = auc_charts(
            gts_speeds,
            probas_speeds,
            auc_chart_labels,
            "driving_decision (speed)",
            "weighted",
            True,
        )
        fig_dir = auc_charts(
            gts_dirs,
            probas_dir,
            auc_chart_labels,
            "driving_decision (dir)",
            "weighted",
            True,
        )
        return results, fig_speed, fig_dir

    return results


def measure_driving_prediction(
    gt_speed: npt.NDArray[np.int_],
    gt_dir: npt.NDArray[np.int_],
    speed_pred: npt.NDArray[np.int_],
    dir_pred: npt.NDArray[np.int_],
    args: DefaultArguments,
    verbose: bool = True,
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
    if verbose:
        print("Evaluating Driving Decision Prediction ...")

    results["speed_Acc"] = accuracy_score(gt_speed, speed_pred)
    results["direction_Acc"] = accuracy_score(gt_dir, dir_pred)

    speed_matrix = confusion_matrix(gt_speed, speed_pred)
    if verbose:
        print("Speed matrix: ", speed_matrix)
    results["speed_confusion_matrix"] = speed_matrix
    sum_cnt = speed_matrix.sum(axis=1)
    sum_cnt = np.array([max(1, i) for i in sum_cnt])
    speed_cls_wise_acc = speed_matrix.diagonal() / sum_cnt
    results["speed_mAcc"] = np.mean(speed_cls_wise_acc)

    dir_matrix = confusion_matrix(gt_dir, dir_pred)
    if verbose:
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
    score, _, _, _ = evaluate_intent(test_gt_file, test_pred_file, args)
    print("Ranking score is : ", score)
