from __future__ import annotations

import json
import os
from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader

from data.prepare_data import get_dataloader
from database.create_database import create_database
from utils.args import DefaultArguments


def main(args: DefaultArguments):
    """Loads the database

    :param DefaultArguments args: The training arguments.
    """
    if not os.path.exists(
        os.path.join(args.database_path, "intent_database_train.pkl")
    ):
        create_database(args)
    else:
        print("Database exists!")
    _, val_loader, test_loader = get_dataloader(args)
    get_intent_gt(val_loader, "../test_gt/val_intent_gt.json", args)
    if test_loader:
        get_intent_gt(test_loader, "../test_gt/test_intent_gt.json", args)


def get_intent_gt(
    dataloader: DataLoader[Any], output_path: str, args: DefaultArguments
) -> None:
    """Gets the ground truth intention and disagreement scores.

    :param torch.utils.data.DataLoader dataloader: The dataloader.
    :param str output_path: The output path.
    :param DefaultArguments args: The training arguments.
    """
    dt = {}
    for itern, data in enumerate(dataloader):
        # if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
        #     gt_intent = data['intention_binary'][:, args.observe_length]
        #     gt_intent_prob = data['intention_prob'][:, args.observe_length]
        # print(data.keys())
        # print(data['frames'])
        for i in range(len(data["frames"])):
            vid = data["video_id"][i]  # str list, bs x 16
            pid = data["ped_id"][i]  # str list, bs x 16
            fid = (
                data["frames"][i][-1] + 1
            ).item()  # int list, bs x 15, observe 0~14, predict 15th intent
            gt_int = data["intention_binary"][i][
                args.observe_length
            ].item()  # int list, bs x 1
            gt_int_prob = data["intention_prob"][i][
                args.observe_length
            ].item()  # float list, bs x 1
            gt_disgr = data["disagree_score"][i][
                args.observe_length
            ].item()  # float list, bs x 1

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]["intent"] = gt_int
            dt[vid][pid][fid]["intent_prob"] = gt_int_prob
            dt[vid][pid][fid]["disagreement"] = gt_disgr

    with open(output_path, "w") as f:
        json.dump(dt, f)


def get_test_driving_gt(
    dataloader: DataLoader[Any],
    output_path: str,
    args: DefaultArguments,
) -> None:
    """Gets the ground truth driving decision.

    :param torch.utils.data.DataLoader dataloader: The dataloader.
    :param DefaultArguments args: The training arguments.
    :param str dset: The dataset.
    """
    dt = {}
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        lbl_speed = data["label_speed"]  # bs x 1
        lbl_dir = data["label_direction"]  # bs x 1
        for i in range(len(data["frames"])):  # for each sample in a batch
            # print(data['video_id'])
            vid = data["video_id"][0][i]  # str list, bs x 60
            fid = (
                data["frames"][i][-1] + 1
            ).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if fid not in dt[vid]:
                dt[vid][fid] = {}
            dt[vid][fid]["speed"] = lbl_speed[i].item()
            dt[vid][fid]["direction"] = lbl_dir[i].item()

        if itern % args.print_freq == 0:
            print(f"Get gt driving decision of Batch {itern}/{niters}")

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dt, f)


def get_intent_reasoning_gt():
    pass
