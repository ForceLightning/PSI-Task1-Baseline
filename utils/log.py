import json
import os
from typing import Any

import numpy as np
from numpy import typing as npt
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from utils.args import DefaultArguments
from utils.metrics import evaluate_driving, evaluate_intent, evaluate_traj
from utils.utils import AverageMeter


class RecordResults:
    def __init__(
        self,
        args: DefaultArguments,
        intent: bool = True,
        traj: bool = True,
        driving: bool = True,
        reason: bool = False,
        evidential: bool = False,
        extract_prediction: bool = False,
    ):
        self.args: DefaultArguments = args
        self.save_output = extract_prediction
        self.intent = intent
        self.traj = traj
        self.driving = driving
        self.reason = reason
        self.evidential = evidential

        self.all_train_results: dict[str, Any] = {}
        self.all_eval_results: dict[str, Any] = {}
        self.all_val_results: dict[str, Any] = {}

        # cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        self.result_path = os.path.join(self.args.checkpoint_path, "results")
        if not os.path.isdir(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)

        self._log_file = os.path.join(self.args.checkpoint_path, "log.txt")
        open(self._log_file, "w").close()

        # self.log_args(self.args)

        # 1. Initalize log info
        # (1.1) Loss log list
        self.log_loss_total: AverageMeter
        self.log_loss_intent: AverageMeter
        self.log_loss_traj: AverageMeter
        self.log_loss_driving_speed: AverageMeter
        self.log_loss_driving_dir: AverageMeter

        # (1.2) Intent
        self.intention_gt: list[npt.NDArray[np.float_]]
        self.intention_prob_gt: list[npt.NDArray[np.float_]]
        self.intention_pred: list[npt.NDArray[np.float_]]
        self.intention_rsn_gt: list[npt.NDArray[np.float_]]
        self.intention_rsn_pred: list[npt.NDArray[np.float_]]

        # (1.2.1) Training data info
        self.frames_list: list[str]
        self.video_list: list[str]
        self.ped_list: list[str]

        # (1.3) Trajectory
        self.traj_gt: list[npt.NDArray[np.float_]]
        self.traj_ori_gt: list[npt.NDArray[np.float_]]
        self.traj_pred: list[npt.NDArray[np.float_]]

        # (1.4) Driving
        self.driving_speed_gt: list[npt.NDArray[np.float_]]
        self.driving_speed_pred: list[npt.NDArray[np.float_]]
        self.driving_dir_gt: list[npt.NDArray[np.float_]]
        self.driving_dir_pred: list[npt.NDArray[np.float_]]

        # (1.5) Store all results
        self.train_epoch_results: dict[str, Any]
        self.eval_epoch_results: dict[str, Any]
        self.epoch: int
        self.nitern: int

    def log_args(self, args: DefaultArguments):
        args_file = os.path.join(self.args.checkpoint_path, "args.txt")
        with open(args_file, "a") as f:
            json.dump(args.__dict__, f, indent=2)
        """ 
            parser = ArgumentParser()
            args = parser.parse_args()
            with open('commandline_args.txt', 'r') as f:
            args.__dict__ = json.load(f)
        """

    def train_epoch_reset(self, epoch: int, nitern: int):
        # 1. initialize log info
        # (1.1) loss log list
        self.log_loss_total = AverageMeter()
        self.log_loss_intent = AverageMeter()
        self.log_loss_traj = AverageMeter()
        self.log_loss_driving_speed = AverageMeter()
        self.log_loss_driving_dir = AverageMeter()
        # (1.2) intent
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        # (1.3) trajectory - args.image_shape
        self.traj_gt = []  # normalized, N x 4, (0, 1) range
        self.traj_ori_gt = []
        self.traj_pred = []  # N x 4 dimension, (0, 1) range
        # (1.4) driving
        self.driving_speed_gt = []
        self.driving_speed_pred = []
        self.driving_dir_gt = []
        self.driving_dir_pred = []
        # (1.5) store all results
        self.train_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

    def train_intent_batch_update(
        self,
        itern: int,
        data: dict[str, torch.Tensor],
        intent_gt: npt.NDArray[np.float_],
        intent_prob_gt: npt.NDArray[np.float_],
        intent_prob: npt.NDArray[np.float_],
        loss: float,
        loss_intent: float,
    ):
        # 3. Update training info
        # (3.1) loss log list
        bs = intent_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_intent.update(loss_intent, bs)
        # (3.2) training data info
        if len(intent_prob) > 0:
            # (3.3) intent
            self.intention_gt.extend(intent_gt)  # bs
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)  # bs

            # assert len(self.intention_gt[0]) == 1 #self.args.predict_length, intent only predict 1 result
        else:
            pass

        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", "a") as f:
                _ = f.write(
                    "Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  Intent Loss: {:.4f} \n".format(
                        self.epoch,
                        self.args.epochs,
                        itern,
                        self.nitern,
                        self.log_loss_total.avg,
                        self.log_loss_intent.avg,
                    )
                )

    def train_intent_epoch_calculate(self, writer: SummaryWriter | None = None):
        intent_results = None
        print("----------- Training results: ------------------------------------ ")
        if self.intention_pred:
            intent_results = evaluate_intent(
                np.array(self.intention_gt),
                np.array(self.intention_prob_gt),
                np.array(self.intention_pred),
                self.args,
            )
            self.train_epoch_results["intent_results"] = intent_results

        print("----------------------------------------------------------- ")
        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename="train")

        # write scalar to tensorboard
        if writer and intent_results:
            for key in ["MSE", "Acc", "F1", "mAcc"]:
                val = intent_results[key]
                writer.add_scalar(f"Train/Results/{key}", val, self.epoch)

            for i in range(self.args.intent_num):
                for j in range(self.args.intent_num):
                    val = intent_results["ConfusionMatrix"][i][j]
                    writer.add_scalar(f"ConfusionMatrix/train{i}_{j}", val, self.epoch)

    def eval_epoch_reset(
        self,
        epoch: int,
        nitern: int,
        intent: bool = True,
        traj: bool = True,
        args: DefaultArguments | None = None,
    ):
        # 1. initialize log info
        self.log_loss_total = AverageMeter()
        self.log_loss_driving_speed = AverageMeter()
        self.log_loss_driving_dir = AverageMeter()
        # (1.2) training data info
        self.frames_list = []
        self.video_list = []
        self.ped_list = []
        # (1.3) intent
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        self.intention_rsn_gt = []
        self.intention_rsn_pred = []

        # (1.4) trajectory - args.image_shape
        self.traj_gt = []  # normalized, N x 4, (0, 1) range
        self.traj_ori_gt = (
            []
        )  # original bboxes before normalization. equal to bboxes if no normalization
        self.traj_pred = []  # N x 4 dimension, (0, 1) range

        # (1.5) driving
        self.driving_speed_gt = []
        self.driving_speed_pred = []
        self.driving_dir_gt = []
        self.driving_dir_pred = []

        # (1.6) store all results
        self.eval_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

    def eval_intent_batch_update(
        self,
        itern: int,
        data: dict[str, Any],
        intent_gt: npt.NDArray[np.float_],
        intent_prob: npt.NDArray[np.float_],
        intent_prob_gt: npt.NDArray[np.float_],
        intent_rsn_gt: npt.NDArray[Any] | None = None,
        intent_rsn_pred: npt.NDArray[Any] | None = None,
    ):
        # 3. Update training info
        # (3.1) loss log list
        bs = intent_gt.shape[0]
        # (3.2) training data info
        self.frames_list.extend(
            data["frames"].detach().cpu().numpy()
        )  # bs x sq_length(60)
        assert len(self.frames_list[0]) == self.args.observe_length
        self.video_list.extend(data["video_id"])  # bs
        self.ped_list.extend(data["ped_id"])
        # print("save record: video list - ", data['video_id'])

        # (3.3) intent
        if len(intent_prob) > 0:
            self.intention_gt.extend(intent_gt)  # bs
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)  # bs
            if intent_rsn_gt is not None:
                self.intention_rsn_gt.extend(intent_rsn_gt)
                self.intention_rsn_pred.extend(intent_rsn_pred)
            # assert len(self.intention_gt[0]) == 1 #self.args.predict_length, intent only predict 1 result
        else:
            pass

    def eval_intent_epoch_calculate(self, writer: SummaryWriter):
        print("----------- Evaluate results: ------------------------------------ ")

        intent_results = None
        if self.intention_pred:
            intent_results = evaluate_intent(
                np.array(self.intention_gt),
                np.array(self.intention_prob_gt),
                np.array(self.intention_pred),
                self.args,
            )
            self.eval_epoch_results["intent_results"] = intent_results

        print(
            "----------------------finished evalcal------------------------------------- "
        )
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename="eval")
        print("log info finished")

        # write scalar to tensorboard
        if writer and intent_results:
            for key in ["MSE", "Acc", "F1", "mAcc"]:
                val = intent_results[key]
                writer.add_scalar(f"Eval/Results/{key}", val, self.epoch)

            for i in range(self.args.intent_num):
                for j in range(self.args.intent_num):
                    val = intent_results["ConfusionMatrix"][i][j]
                    writer.add_scalar(f"ConfusionMatrix/eval{i}_{j}", val, self.epoch)

    # def save_results(self, prefix=''):
    #     self.result_path = os.path.join(self.args.checkpoint_path, 'results', f'epoch_{self.epoch}', prefix)
    #     if not os.path.isdir(self.result_path):
    #         os.makedirs(self.result_path)
    #     # 1. train results
    #     np.save(self.result_path + "/train_results.npy", self.all_train_results)
    #     # 2. eval results
    #     np.save(self.result_path + "/eval_results.npy", self.all_eval_results)
    #
    #     # 3. save data
    #     np.save(self.result_path + "/intent_gt.npy", self.intention_gt)
    #     np.save(self.result_path + "/intent_prob_gt.npy", self.intention_prob_gt)
    #     np.save(self.result_path + "/intent_pred.npy", self.intention_pred)
    #     np.save(self.result_path + "/frames_list.npy", self.frames_list)
    #     np.save(self.result_path + "/video_list.npy", self.video_list)
    #     np.save(self.result_path + "/ped_list.npy", self.ped_list)
    #     np.save(self.result_path + "/intent_rsn_gt.npy", self.intention_rsn_gt)
    #     np.save(self.result_path + "/intent_rsn_pred.npy", self.intention_rsn_pred)
    #

    # 3. Update traj training info
    def train_traj_batch_update(
        self,
        itern: int,
        data: dict[str, torch.Tensor],
        traj_gt: npt.NDArray[np.float_],
        traj_pred: npt.NDArray[np.float_],
        loss: float,
        loss_traj: float,
    ):
        # evidence: bs x ts x 4: mu,v,alpha,beta

        # (3.1) loss log list
        bs, _, _ = traj_gt.shape  # bs x 45 x 4
        self.log_loss_total.update(loss, bs)
        self.log_loss_traj.update(loss_traj, bs)
        # (3.2) training data info
        if traj_pred.size != 0:
            self.traj_gt.extend(traj_gt)  # bs x pred_seq(45) x 4
            self.traj_pred.extend(traj_pred)  # bs x pred_seq(45) x 4
        else:
            pass

        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", "a") as f:
                _ = f.write(
                    "Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  Intent Loss: {:.4f} |  Traj Loss: {:.4f}\n".format(
                        self.epoch,
                        self.args.epochs,
                        itern,
                        self.nitern,
                        self.log_loss_total.avg,
                        self.log_loss_intent.avg,
                        self.log_loss_traj.avg,
                    )
                )

    def train_traj_epoch_calculate(self, writer: SummaryWriter | None = None):
        print("----------- Training results: ------------------------------------ ")
        traj_results = None
        if self.traj_pred != []:
            traj_results = evaluate_traj(
                np.array(self.traj_gt), np.array(self.traj_pred), self.args
            )
            self.train_epoch_results["traj_results"] = traj_results
        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results

        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename="train")
        # write scalar to tensorboard
        if writer and traj_results:
            for key in [
                "ADE",
                "FDE",
                "ARB",
                "FRB",
            ]:  # , 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
                for time in ["0.5", "1.0", "1.5"]:
                    val = traj_results[key][time]
                    writer.add_scalar(f"Train/Results/{key}_{time}", val, self.epoch)

    def eval_traj_batch_update(
        self,
        itern: int,
        data: dict[str, torch.Tensor | list[str | float | int]],
        traj_gt: npt.NDArray[np.float_],
        traj_pred: npt.NDArray[np.float_],
    ):
        # 3. Update training info
        self.frames_list.extend(
            data["frames"].detach().cpu().numpy()
        )  # bs x sq_length(60)
        assert len(self.frames_list[0]) == self.args.observe_length
        self.video_list.extend(data["video_id"])  # bs
        self.ped_list.extend(data["ped_id"])
        # (3.1) loss log list
        # bs, ts, dim = traj_gt.shape  # bs x 45 x 4
        self.traj_ori_gt.extend(data["original_bboxes"].detach().cpu().numpy())

        if traj_pred.size != 0:
            self.traj_gt.extend(traj_gt)  # bs x pred_seq(45) x 4
            self.traj_pred.extend(traj_pred)  # bs x pred_seq(45) x 4
            assert len(self.traj_gt[0]) == self.args.predict_length
        else:
            pass

    def eval_traj_epoch_calculate(self, writer: SummaryWriter):
        print("----------- Eval results: ------------------------------------ ")
        traj_results = None
        if self.traj_pred != []:
            traj_results = evaluate_traj(
                np.array(self.traj_gt), np.array(self.traj_pred), self.args
            )
            self.eval_epoch_results["traj_results"] = traj_results

        # Update epoch to all results
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        # self.log_msg(msg='Epoch {} \n --------------------------'.format(self.epoch), filename='train_results.txt')
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename="eval")
        # write scalar to tensorboard
        if writer and traj_results:
            for key in [
                "ADE",
                "FDE",
                "ARB",
                "FRB",
            ]:  # , 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
                for time in ["0.5", "1.0", "1.5"]:
                    val = traj_results[key][time]
                    writer.add_scalar(f"Eval/Results/{key}_{time}", val, self.epoch)
                    print(f"Epoch {self.epoch}: {key}_{time}", val)
        print("----------------------------------------------------------- ")

    def train_driving_batch_update(
        self,
        itern: int,
        data: dict[str, torch.Tensor | list[str | int | float]],
        speed_gt: npt.NDArray[np.int_],
        direction_gt: npt.NDArray[np.int_],
        speed_pred_logit: npt.NDArray[np.float_],
        dir_pred_logit: npt.NDArray[np.float_],
        loss: float,
        loss_driving_speed: float,
        loss_driving_dir: float,
    ) -> None:
        # 3. Update training info
        # (3.1) loss log list
        bs = speed_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_driving_speed.update(loss_driving_speed, bs)
        self.log_loss_driving_dir.update(loss_driving_dir, bs)
        # (3.2) training data info

        self.driving_speed_gt.extend(speed_gt)  # bs
        self.driving_dir_gt.extend(direction_gt)
        self.driving_speed_pred.extend(np.argmax(speed_pred_logit, axis=-1))  # bs
        self.driving_dir_pred.extend(np.argmax(dir_pred_logit, axis=-1))  # bs

        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", "a") as f:
                _ = f.write(
                    "Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  driving speed Loss: {:.4f} |  driving dir Loss: {:.4f} \n".format(
                        self.epoch,
                        self.args.epochs,
                        itern,
                        self.nitern,
                        self.log_loss_total.avg,
                        self.log_loss_driving_speed.avg,
                        self.log_loss_driving_dir.avg,
                    )
                )

    def train_driving_epoch_calculate(
        self, writer: SummaryWriter | None = None
    ) -> None:
        print("----------- Training results: ------------------------------------ ")
        driving_results = None
        if self.driving:
            driving_results = evaluate_driving(
                np.array(self.driving_speed_gt),
                np.array(self.driving_dir_gt),
                np.array(self.driving_speed_pred),
                np.array(self.driving_dir_pred),
                self.args,
            )
            self.train_epoch_results["driving_results"] = driving_results
            # {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}

        # Update epoch to all results
        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename="train")

        # write scalar to tensorboard
        if writer and driving_results:
            for key in [
                "speed_Acc",
                "speed_mAcc",
                "direction_Acc",
                "direction_mAcc",
            ]:  # driving_results.keys(): #
                if key not in driving_results.keys():
                    continue
                val = driving_results[key]
                print("results: ", key, val)
                writer.add_scalar(f"Train/Results/{key}", val, self.epoch)
        print("----------------------------------------------------------- ")

    def eval_driving_batch_update(
        self,
        itern: int,
        data: dict[str, torch.Tensor | list[float | int | str]],
        speed_gt: npt.NDArray[np.float_],
        direction_gt: npt.NDArray[np.float_],
        speed_pred_logit: npt.NDArray[np.float_],
        dir_pred_logit: npt.NDArray[np.float_],
        reason_gt=None,
        reason_pred=None,
    ) -> None:
        # 3. Update training info
        # (3.1) loss log list
        # bs = speed_gt.shape[0]
        # self.frames_list.extend(data['frames'].detach().cpu().numpy())  # bs x sq_length(60)
        # assert len(self.frames_list[0]) == self.args.observe_length
        # self.video_list.extend(data['video_id'])  # bs
        # (3.2) training data info

        self.driving_speed_gt.extend(speed_gt)  # bs
        self.driving_dir_gt.extend(direction_gt)
        self.driving_speed_pred.extend(np.argmax(speed_pred_logit, axis=-1))  # bs
        self.driving_dir_pred.extend(np.argmax(dir_pred_logit, axis=-1))  # bs
        if reason_pred is not None:
            pass  # store reason prediction

    def eval_driving_epoch_calculate(self, writer: SummaryWriter | None = None) -> None:
        print("----------- Evaluate results: ------------------------------------ ")
        driving_results = None
        if self.driving:
            driving_results = evaluate_driving(
                np.array(self.driving_speed_gt),
                np.array(self.driving_dir_gt),
                np.array(self.driving_speed_pred),
                np.array(self.driving_dir_pred),
                self.args,
            )
            self.eval_epoch_results["driving_results"] = driving_results
            # {'speed_Acc': 0, 'speed_mAcc': 0, 'direction_Acc': 0, 'direction_mAcc': 0}
            for key in self.eval_epoch_results["driving_results"].keys():
                print(key, self.eval_epoch_results["driving_results"][key])
        # Update epoch to all results
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename="eval")

        # write scalar to tensorboard
        if writer and driving_results:
            for key in ["speed_Acc", "speed_mAcc", "direction_Acc", "direction_mAcc"]:
                if key not in driving_results.keys():
                    continue
                val = driving_results[key]
                print("results: ", key, val)
                writer.add_scalar(f"Eval/Results/{key}", val, self.epoch)
        print("log info finished")
        print(
            "----------------------finished results calculation------------------------------------- "
        )

    def log_msg(self, msg: str, filename: str | None = None):
        if not filename:
            filename = os.path.join(self.args.checkpoint_path, "log.txt")
        else:
            pass
        savet_to_file = filename
        with open(savet_to_file, "a") as f:
            _ = f.write(str(msg) + "\n")

    def log_info(self, epoch: int, info: dict[str, Any], filename: str | None = None):
        if not filename:
            filename = "log.txt"
        else:
            pass
        for key in info:
            savet_to_file = os.path.join(
                self.args.checkpoint_path, filename + "_" + key + ".txt"
            )
            self.log_msg(
                msg="Epoch {} \n --------------------------".format(epoch),
                filename=savet_to_file,
            )
            with open(savet_to_file, "a") as f:
                if isinstance(info[key], str):
                    _ = f.write(info[key] + "\n")
                elif isinstance(info[key], dict):
                    for k in info[key]:
                        _ = f.write(k + ": " + str(info[key][k]) + "\n")
                else:
                    _ = f.write(str(info[key]) + "\n")
            self.log_msg(
                msg=".................................................".format(
                    self.epoch
                ),
                filename=savet_to_file,
            )
