from datetime import datetime
import json
import os
import pickle

from test import test_intent, validate_intent
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from opts import get_opts
from train import train_intent
from utils.log import RecordResults


def main(args):
    writer = SummaryWriter(args.checkpoint_path)
    recorder = RecordResults(args)
    """ 1. Load database """
    if not os.path.exists(
        os.path.join(args.database_path, "intent_database_train.pkl")
    ):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)
    get_intent_gt(val_loader, "../test_gt/val_intent_gt.json", args)
    get_intent_gt(test_loader, "../test_gt/test_intent_gt.json", args)


def get_intent_gt(dataloader, output_path, args):
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


def get_test_driving_gt(model, dataloader, args, dset="test"):
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

        # if itern >= 10:
        #     break
    with open(os.path.join(f"./test_gt/{dset}_driving_gt.json"), "w") as f:
        json.dump(dt, f)


def get_intent_reasoning_gt():
    pass
