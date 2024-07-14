import argparse
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.evaluate_results import evaluate_drivings, evaluate_intents, evaluate_trajs


def main(opts: argparse.Namespace) -> None:
    sns.set_theme("paper", "whitegrid")
    pred_fn: str = ""
    test_gt_fn: str = opts.gt_dir
    match opts.task_name:
        case "ped_intent":
            pred_fn = os.path.join("results", "test_intent_pred.json")
            test_gt_fn = os.path.join(opts.gt_dir, "test_intent_gt.json")
        case "ped_traj":
            pred_fn = os.path.join("results", "test_traj_pred.json")
            test_gt_fn = os.path.join(opts.gt_dir, "test_traj_gt.json")

        case "driving_decision":
            pred_fn = os.path.join("results", "test_driving_pred.json")
            test_gt_fn = os.path.join(opts.gt_dir, "test_driving_gt.json")

    if not os.path.exists(test_gt_fn):
        raise FileNotFoundError(f"Ground truth file not found: {test_gt_fn}")

    gt_paths = [test_gt_fn] * len(opts.checkpoint_path)
    pred_paths: list[str] = []
    model_names: list[str] = []
    for checkpoint_path in opts.checkpoint_path:
        pred_paths.append(os.path.join(checkpoint_path, pred_fn))
        model_names.append(
            os.path.normpath(checkpoint_path).split(os.sep)[-2]
        )  # get the model name from the checkpoint path

    paths = list(zip(gt_paths, pred_paths))
    match opts.task_name:
        case "ped_intent":
            results, fig = evaluate_intents(
                paths,
                True,
                model_names,
            )
            _ = fig.savefig("intent_results.png", transparent=True)
            plt.show()
        case "ped_traj":
            results = evaluate_trajs(paths)
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.expand_frame_repr",
                False,
            ):
                print(results)
        case "driving_decision":
            results, fig_speed, fig_dir = evaluate_drivings(paths, True, model_names)
            _ = fig_speed.savefig("speed_results.png", transparent=True)
            _ = fig_dir.savefig("dir_results.png", transparent=True)
            plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("compare model evaluation")
    _ = arg_parser.add_argument(
        "-p",
        "--checkpoint_path",
        metavar="P",
        type=str,
        nargs="+",
        help="[p]ath to a checkpoint directory",
    )

    _ = arg_parser.add_argument(
        "-t",
        "--task_name",
        type=str,
        default="ped_intent",
        choices=["ped_intent", "ped_traj", "driving_decision"],
        help="[t]ask name",
    )

    _ = arg_parser.add_argument(
        "-g",
        "--gt_dir",
        type=str,
        default="../test_gt/",
        help="path to [g]round truth directory",
    )

    opts = arg_parser.parse_args()
    main(opts)
