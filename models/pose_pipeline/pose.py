from __future__ import annotations

import os
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from boxmot import BoTSORT, BYTETracker, DeepOCSORT
from boxmot.trackers.basetracker import BaseTracker
from cv2 import typing as cvt
from numpy import typing as npt
from torch import nn
from torch.utils.data import default_collate
from tqdm.auto import tqdm
from typing_extensions import overload, override
from ultralytics import YOLO
from ultralytics.engine.results import Results

from data.custom_dataset import T_intentBatch, T_intentSample
from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from models.traj_modules.model_transformer_traj_bbox import TransformerTrajBbox
from opts import get_opts
from utils.args import DefaultArguments
from utils.cuda import *


class YOLOWithTracker:
    """Class that wraps a YOLO model and a tracker together.

    :param YOLO yolo_model: Instantiated YOLO model.
    :tracker BaseTracker tracker: Instantiated tracker.
    """

    def __init__(self, yolo_model: YOLO, tracker: BaseTracker):
        super().__init__()
        self.yolo_model = yolo_model
        self.tracker = tracker

    def _predict_on_collection(
        self, images: list[npt.NDArray[np.uint8] | cvt.MatLike]
    ) -> tuple[npt.NDArray[np.uint8], list[Results]]:
        results: list[Results] = self.yolo_model.predict(
            images, verbose=False, classes=[0], conf=0.15
        )
        ret_images: npt.NDArray[np.uint8] = np.asarray(images)
        return ret_images, results

    def _predict_on_numpy(
        self, images: npt.NDArray[np.uint8] | cvt.MatLike
    ) -> tuple[npt.NDArray[np.uint8] | cvt.MatLike, list[Results]]:
        results: list[Results] = self.yolo_model.predict(
            images, verbose=False, classes=[0], conf=0.15
        )
        return images, results

    def _predict_on_tensor(
        self, images: torch.Tensor
    ) -> tuple[npt.NDArray[np.uint8] | cvt.MatLike, list[Results]]:
        np_images: npt.NDArray[np.uint8] = (
            images.permute((0, 2, 3, 1)).detach().cpu().numpy()
        )
        np_images = (np_images.squeeze() * 255.0).astype(np.uint8)

        results: list[Results] = self.yolo_model.predict(
            images, verbose=False, classes=[0], conf=0.15
        )
        return np_images, results

    @overload
    def __call__(
        self, images: torch.Tensor
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...

    @overload
    def __call__(
        self, images: list[npt.NDArray[np.uint8] | cvt.MatLike]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...

    @overload
    def __call__(
        self, images: npt.NDArray[np.uint8] | cvt.MatLike
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...

    def __call__(
        self,
        images: (
            torch.Tensor
            | list[npt.NDArray[np.uint8] | cvt.MatLike]
            | npt.NDArray[np.uint8]
            | cvt.MatLike
        ),
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Performs pose estimation and tracking on a batch of images or a single image.

        :param images: Batch of images of shape B x C x H x W
        :type images: torch.Tensor or list[npt.NDArray[np.uint8] or cvt.MatLike] or
        npt.NDArray[np.uint8] or cvt.MatLike
        :return: Dictionary of track IDs and tuple of (bounding boxes, keypoints,
        keypoint observation mask).
        :rtype: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        """
        # Reshape numpy images back to cv2 RGB format B x H x W x C
        results: list[Results] = []
        np_images: npt.NDArray[np.uint8] | cvt.MatLike
        if isinstance(images, torch.Tensor):
            np_images, results = self._predict_on_tensor(images)
        elif isinstance(images, list):
            np_images, results = self._predict_on_collection(images)
        else:
            np_images, results = self._predict_on_numpy(images)

        ts = np_images.shape[0]

        # NOTE: This is a bit of a hacky way to get the tracker to work with the
        # YOLO detections. It may be better to refactor the tracker to work with
        # the YOLO detections directly.

        # {track_id: {frame_id: det_ind}}
        tracked_map: dict[int, dict[int, int]] = {}

        # NOTE: It may be faster to provide YOLO with a batched input then iterate over
        # the results later.

        for i, image in enumerate(np_images):
            dets: list[list[float | int]] = []
            kp_dets: list[list[float]] = []
            result = results[i]

            if (keypoints := result.keypoints) is not None:
                for k in keypoints:
                    kp = k.data
                    kp_dets.append(kp.tolist())
            else:
                print(f"{i:02d}: No keypoints detected.")
            #     continue

            # TODO(chris): Propagate the confidence score out of this method.
            if (boxes := result.boxes) is not None:
                for box in boxes:
                    box_cls = int(box.cls[0].item())
                    coords: list[float] = box.xyxy[0].tolist()
                    conf: float = box.conf[0].item()

                    dets.append([*coords] + [conf, box_cls])
            else:
                print(f"{i:02d}: No bounding boxes detected.")
            #     continue

            np_dets = np.array(dets)
            np_kp_dets = np.array(kp_dets)

            if len(np_dets) == 0:
                np_dets = np.empty((0, 6))

            tracks: npt.NDArray[np.int_ | np.float_] = self.tracker.update(
                np_dets, image
            )

            if tracks.shape[0] != 0:
                trk_ids = tracks[:, 4].astype("int")
                det_inds = tracks[:, 7].astype("int")
                for trk_id, det_ind in zip(trk_ids, det_inds):
                    if (id_list := tracked_map.get(trk_id)) is not None:
                        id_list[i] = det_ind
                    else:
                        tracked_map[trk_id] = {i: det_ind}

        ret: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        """Format is track_id: (bboxes, poses, pose masks)
        """

        for trk_id, det_inds in tracked_map.items():
            # NOTE: Let the batch size (of 1) be added as a dimension when `collate_fn`
            # does its thing later
            trk_dets_base: torch.Tensor = (
                torch.zeros((ts, 4)).type(FloatTensor).to(DEVICE)
            )
            trk_kp_dets_base: torch.Tensor = (
                torch.zeros((ts, 17, 2)).type(FloatTensor).to(DEVICE)
            )
            trk_kp_dets_mask_base: torch.Tensor = (
                torch.zeros((ts, 17)).type(LongTensor).to(DEVICE)
            )

            for frame_id in range(ts):
                if (det_ind := det_inds.get(frame_id)) is not None:
                    # GUARD
                    if (
                        results[frame_id].boxes is None
                        or results[frame_id].keypoints is None
                    ):
                        continue
                    trk_dets_base[frame_id, :] = (
                        results[frame_id].boxes[det_ind].xyxyn[0]
                    )
                    trk_kp_dets_base[frame_id, :, :] = (
                        results[frame_id].keypoints[det_ind].xy
                    )
                    trk_kp_dets_mask_base[frame_id, :] = 1

            ret[trk_id] = (
                trk_dets_base,
                trk_kp_dets_base,
                trk_kp_dets_mask_base.bool(),
            )

        return ret

    def reset_tracker(self):
        # WARNING: this may break things.
        self.tracker.active_tracks = []
        self.tracker.frame_count = 0


@dataclass
class YOLOPipelineResults:
    bboxes: torch.Tensor
    poses: torch.Tensor | None


class YOLOTrackerPipelineModel(nn.Module):
    def __init__(
        self,
        args: DefaultArguments,
        model: nn.Module,
        yolo_tracker: YOLOWithTracker,
        return_dict: bool = False,
        return_pose: bool = False,
    ):
        super().__init__()
        self.args = args
        self.model = model
        self.yolo_tracker = yolo_tracker
        self.return_dict = return_dict
        self.return_pose = return_pose
        if self.return_pose and not self.return_dict:
            raise ValueError(
                "return_pose is only valid when return_dict is set to True."
            )

    def _tracker_infer_intent_batch(
        self, x: T_intentBatch
    ) -> tuple[
        torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ]:
        imgs = x["image"]
        assert isinstance(imgs, torch.Tensor), "input images must not be an empty list"
        imgs = imgs.squeeze()
        assert (
            imgs.dim() == 4
        ), f"input dimensionality must be 4, but is of shape {imgs.shape}"

        self.yolo_tracker.reset_tracker()
        tracks = self.yolo_tracker(imgs)
        return imgs, tracks

    def _tracker_infer_intent_ndarray(self, x: npt.NDArray[np.uint8]) -> tuple[
        npt.NDArray[np.uint8],
        dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        self.yolo_tracker.reset_tracker()
        im: list[npt.NDArray[np.uint8]] = [i for i in x]
        tracks = self.yolo_tracker(im)
        return x, tracks

    def _tracker_infer_intent_deque(self, x: deque[cvt.MatLike]) -> tuple[
        npt.NDArray[np.uint8],
        dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        self.yolo_tracker.reset_tracker()
        im: list[npt.NDArray[np.uint8]] = [np.asarray(i) for i in x]
        tracks = self.yolo_tracker(im)
        return np.asarray(im), tracks

    @override
    def forward(self, x: T_intentBatch) -> torch.Tensor | YOLOPipelineResults:
        tracks: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        imgs, tracks = self._tracker_infer_intent_batch(x)

        ts = imgs.shape[0]

        if len(tracks.keys()) == 0:
            preds = torch.zeros((1, self.args.max_track_size, 4), dtype=torch.float16)
            if self.return_dict:
                return YOLOPipelineResults(preds, None)
            return preds

        samples: list[T_intentSample] = []
        # TODO(chris): Take the detected kps and boxes and convert to a format that the
        # inner prediction model can parse.

        # construct the batch for input to the prediction head.
        for track_id, (boxes, poses, pose_masks) in tracks.items():
            sample: T_intentSample = {
                "image": imgs,
                "pose": poses,
                "bboxes": boxes,
                "frames": x["frames"],
                "ped_id": [f"track_{track_id:03d}"] * ts,
                "video_id": x["video_id"],
                "total_frames": x["total_frames"],
                "pose_masks": pose_masks,
                "local_featmaps": x["local_featmaps"],
                "intention_prob": np.zeros((ts), dtype=np.float_),
                "intention_binary": np.zeros((ts), dtype=np.int_),
                "disagree_score": np.zeros((ts), dtype=np.float_),
                "global_featmaps": x["global_featmaps"],
                "original_bboxes": boxes,
            }
            samples.append(sample)

        batch_tensor: T_intentBatch = default_collate(samples)
        preds: torch.Tensor

        # NOTE: Pose-based TS-Transformers are kinda broken with this implementation
        # as they require poses from future timesteps to make a prediction.
        if isinstance(self.model, TransformerTrajBbox):
            preds = self.model.generate(batch_tensor)
        else:
            preds = self.model(batch_tensor)

        past_future_boxes: torch.Tensor = torch.cat(
            (batch_tensor["bboxes"], preds), dim=1
        )

        if self.return_dict:
            if self.return_pose:
                return YOLOPipelineResults(past_future_boxes, batch_tensor["pose"])
            return YOLOPipelineResults(past_future_boxes, None)

        return preds


def main(args: DefaultArguments):
    args.batch_size = 1
    args.database_file = "traj_database_train.pkl"
    args.intent_model = False
    args.traj_model = True
    args.traj_loss = ["nll"]
    args.model_name = "tcn_traj_bbox"
    args.loss_weights = {
        "loss_intent": 0.0,
        "loss_traj": 1.0,
        "loss_driving": 0.0,
    }
    args.load_image = True
    args.backbone = ""
    args.freeze_backbone = False
    args.seq_overlap_rate = 1
    args.test_seq_overlap_rate = 1
    args.observe_length = 15
    args.predict_length = 45
    args.max_track_size = args.observe_length + args.predict_length

    if "transformer" in args.model_name:
        args.observe_length += 1
        args.max_track_size += 1

    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)

    train_loader, _, _ = get_dataloader(
        args, shuffle_train=False, load_test=False, yolo_transforms=False
    )
    args.steps_per_epoch = len(train_loader.dataset)

    model, _, _ = build_model(args)
    yolo_model = YOLO("yolov8s-pose.pt")
    tracker = DeepOCSORT(
        model_weights=Path("osnet_x0_25_msmt17.pt"), device="cuda:0", fp16=True
    )

    yolo_tracker = YOLOWithTracker(yolo_model, tracker)
    pipeline_model = YOLOTrackerPipelineModel(args, model, yolo_tracker)

    dl_iterator = tqdm(
        enumerate(train_loader),
        total=args.steps_per_epoch,
        desc="Batches",
        position=0,
        leave=True,
    )
    for i, data in dl_iterator:
        preds = pipeline_model(data)
        dl_iterator.write(str(preds))


if __name__ == "__main__":
    args = get_opts()
    main(args)
