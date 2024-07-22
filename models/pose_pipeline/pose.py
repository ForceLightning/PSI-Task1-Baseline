from __future__ import annotations

import abc
import ast
import os
from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from boxmot import BoTSORT, BYTETracker, DeepOCSORT
from boxmot.trackers.basetracker import BaseTracker
from cv2 import typing as cvt
from numpy import typing as npt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from tqdm.auto import tqdm
from typing_extensions import override
from ultralytics import YOLO
from ultralytics.engine.results import Results

from data.custom_dataset import T_intentBatch, T_intentSample
from data.prepare_data import get_dataloader
from database.create_database import T_intentDB, create_database
from misc.utils import find_person_id_associations
from models.build_model import build_model
from models.model_interfaces import ITSTransformerWrapper
from opts import get_opts
from SimpleHRNet import SimpleHRNet
from utils.args import DefaultArguments
from utils.cuda import *


class TrackerWrapper(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(
        self, image: torch.Tensor | npt.NDArray[np.uint8] | cvt.MatLike
    ) -> dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Predicts bounding boxes and pose keypoints on an image and tracks them.

        :param image: Singular frame to predict on.
        :type image: torch.Tensor | npt.NDArray[np.uint8] | cvt.MatLike
        :return: Dictionary of track id: (bboxes, poses, pose masks, bbox confidence,
        pose confidence)
        :rtype: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor]]
        """
        pass

    @abc.abstractmethod
    def reset_tracker(self) -> None:
        """Resets the tracker to its initial state."""
        pass


class PipelineWrapper(nn.Module, metaclass=abc.ABCMeta):
    model: nn.Module
    tracker: TrackerWrapper
    return_dict: bool = False
    return_pose: bool = False

    @abc.abstractmethod
    def get_imgs_and_tracks(
        self,
        x: T_intentBatch,
    ) -> tuple[
        npt.NDArray[np.uint8] | torch.Tensor,
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        pass

    @abc.abstractmethod
    def generate_samples(
        self,
        imgs: npt.NDArray[np.uint8] | torch.Tensor,
        tracks: dict[int, tuple[torch.Tensor, ...]],
        x: T_intentBatch,
    ) -> list[T_intentSample]:
        pass

    @abc.abstractmethod
    def pack_preds(
        self,
        x: T_intentBatch,
        preds: torch.Tensor,
    ) -> torch.Tensor | PipelineResults | None:
        pass

    @torch.no_grad()
    def forward(self, x: T_intentBatch) -> torch.Tensor | PipelineResults | None:
        tracks: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        imgs, tracks = self.get_imgs_and_tracks(x)

        preds: torch.Tensor

        if len(tracks.keys()) == 0:
            return None

        samples: list[T_intentSample] = self.generate_samples(imgs, tracks, x)

        if len(samples) == 0:
            return None

        batch_tensor: T_intentBatch = default_collate(samples)

        model_instance = getattr(self.model, "module", self.model)

        if isinstance(model_instance, ITSTransformerWrapper):
            preds = model_instance.generate(batch_tensor)
        else:
            preds = model_instance(batch_tensor)
        return self.pack_preds(batch_tensor, preds)


class YOLOTrackerWrapper(TrackerWrapper):
    """Class that wraps a YOLO model and a tracker together.

    :param YOLO yolo_model: Instantiated YOLO model.
    :tracker BaseTracker tracker: Instantiated tracker.
    """

    def __init__(
        self,
        args: DefaultArguments,
        yolo_model: YOLO,
        tracker: BaseTracker,
        persistent_tracker: bool = False,
    ):
        super().__init__()
        self.yolo_model = yolo_model
        self.tracker = tracker
        self.persistent_tracker = persistent_tracker
        self.frame_cache: deque[cvt.MatLike | npt.NDArray[np.uint8] | torch.Tensor] = (
            deque(maxlen=args.observe_length)
        )
        self.results_cache: list[Results] = deque(maxlen=args.observe_length)

        # {track_id: {frame_id: det_ind}}
        self.tracked_map: dict[int, dict[int, int]] = {}
        self.frame_count = 0

    def _track(
        self, image: npt.NDArray[np.uint8] | cvt.MatLike, result: Results
    ) -> dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # NOTE: This is a bit of a hacky way to get the tracker to work with the
        # YOLO detections. It may be better to refactor the tracker to work with
        # the YOLO detections directly.

        ts = len(self.frame_cache)

        assert image.ndim == 3, "Input must be of shape (H, W, C)."

        self.frame_count += 1

        dets: list[list[float | int]] = []
        kp_dets: list[list[float]] = []

        if (keypoints := result.keypoints) is not None:
            for k in keypoints:
                kp = k.data
                kp_dets.append(kp.tolist())
        else:
            print(f"{self.frame_count:02d}: No keypoints detected.")
        #     continue

        if (boxes := result.boxes) is not None:
            for box in boxes:
                box_cls = int(box.cls[0].item())
                coords: list[float] = box.xyxy[0].tolist()
                conf: float = box.conf[0].item()

                dets.append([*coords] + [conf, box_cls])
        else:
            print(f"{self.frame_count:02d}: No bounding boxes detected.")
        #     continue

        np_dets = np.array(dets)
        np_kp_dets = np.array(kp_dets)

        if len(np_dets) == 0:
            np_dets = np.empty((0, 6))

        tracks: npt.NDArray[np.int_ | np.float_] = self.tracker.update(np_dets, image)

        if tracks.shape[0] != 0:
            trk_ids = tracks[:, 4].astype("int")
            det_inds = tracks[:, 7].astype("int")
            for trk_id, det_ind in zip(trk_ids, det_inds):
                if (id_list := self.tracked_map.get(trk_id)) is not None:
                    id_list[self.frame_count] = det_ind
                else:
                    self.tracked_map[trk_id] = {self.frame_count: det_ind}

        ret: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        """Format is track_id: (bboxes, poses, pose masks, bbox confidence)
        """

        for trk_id, det_inds in self.tracked_map.items():
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
            trk_dets_conf_base: torch.Tensor = (
                torch.zeros((ts, 1)).type(FloatTensor).to(DEVICE)
            )
            trk_kp_dets_conf_base: torch.Tensor = (
                torch.zeros((ts, 17)).type(FloatTensor).to(DEVICE)
            )

            for frame_id in range(self.frame_count - ts, self.frame_count):
                relative_frame_id = frame_id - (self.frame_count - ts)
                if (det_ind := det_inds.get(frame_id)) is not None:
                    # GUARD
                    if (
                        self.results_cache[relative_frame_id].boxes is None
                        or self.results_cache[relative_frame_id].keypoints is None
                    ):
                        continue
                    try:
                        trk_dets_base[relative_frame_id, :] = (
                            self.results_cache[relative_frame_id]
                            .boxes[det_ind]
                            .xyxyn[0]
                        )
                        trk_dets_conf_base[relative_frame_id, :] = (
                            self.results_cache[relative_frame_id].boxes[det_ind].conf[0]
                        )
                    except IndexError:
                        pass

                    try:
                        trk_kp_dets_base[relative_frame_id, :, :] = (
                            self.results_cache[relative_frame_id].keypoints[det_ind].xy
                        )
                        trk_kp_dets_conf_base[relative_frame_id, :] = (
                            self.results_cache[relative_frame_id]
                            .keypoints[det_ind]
                            .conf
                        )
                    except IndexError:
                        pass
                    except RuntimeError:
                        # print(
                        #     trk_kp_dets_base.shape,
                        #     self.results_cache[relative_frame_id]
                        #     .keypoints[det_ind]
                        #     .xy.shape,
                        # )
                        # NOTE: Fail silently, this just means that the keypoints are
                        # not detected
                        pass

                    trk_kp_dets_mask_base[relative_frame_id, :] = 1
                else:
                    # print(
                    #     f"Frame {self.frame_count:02d} has no detection for track {trk_id}"
                    # )
                    # NOTE: Fail silently, this just means that the track is no longer
                    # available in the results cache.
                    pass

            ret[trk_id] = (
                trk_dets_base,
                trk_kp_dets_base,
                trk_kp_dets_mask_base.bool(),
                trk_dets_conf_base,
                trk_kp_dets_conf_base,
            )

        return ret

    @override
    def __call__(
        self,
        image: torch.Tensor | npt.NDArray[np.uint8] | cvt.MatLike,
    ) -> dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        np_image: npt.NDArray[np.uint8]
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Input must be a 4D tensor."
            assert image.shape[0] == 1, "Batch size must be 1."
            np_image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            np_image = (np_image * 255.0).astype(np.uint8)
        else:
            assert image.ndim == 3, "Input must be of shape (H, W, C)."
            np_image = image

        results: Results = self.yolo_model.predict(
            image, verbose=False, classes=[0], conf=0.15
        )[0]

        if self.persistent_tracker:
            self.results_cache.append(results)
            self.frame_cache.append(np_image)

        return self._track(np_image, results)

    def reset_tracker(self):
        # WARNING: this may break things.
        self.tracker.active_tracks = []
        self.tracker.frame_count = 0


@dataclass
class PipelineResults:
    bboxes: torch.Tensor
    bbox_conf: torch.Tensor | npt.NDArray[np.float_] | None = None
    poses: torch.Tensor | None = None
    pose_conf: torch.Tensor | npt.NDArray[np.float_] | None = None
    intent: torch.Tensor | None = None


class YOLOPipelineModelWrapper(PipelineWrapper):
    def __init__(
        self,
        args: DefaultArguments,
        model: nn.Module,
        yolo_tracker: YOLOTrackerWrapper,
        return_dict: bool = False,
        return_pose: bool = False,
        persistent_tracker: bool = False,
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
        self.persistent_tracker = persistent_tracker
        self.frame_cache: deque[cvt.MatLike | npt.NDArray[np.uint8] | torch.Tensor] = (
            deque(maxlen=args.observe_length)
        )

    def _tracker_infer_intent_batch(self, x: T_intentBatch) -> tuple[
        torch.Tensor,
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        imgs = x["image"]
        assert isinstance(imgs, torch.Tensor), "input images must not be an empty list"
        imgs = imgs.squeeze()
        assert (
            imgs.dim() == 4
        ), f"input dimensionality must be 4, but is of shape {imgs.shape}"

        tracks = self.yolo_tracker(imgs[-1].unsqueeze(0))
        return imgs, tracks

    def _tracker_infer_intent_ndarray(self, x: npt.NDArray[np.uint8]) -> tuple[
        npt.NDArray[np.uint8],
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        if not self.persistent_tracker:
            self.yolo_tracker.reset_tracker()
        tracks: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = self.yolo_tracker(x[-1])
        return x, tracks

    def _tracker_infer_intent_deque(self, x: deque[cvt.MatLike]) -> tuple[
        npt.NDArray[np.uint8],
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        if not self.persistent_tracker:
            self.yolo_tracker.reset_tracker()
        im: list[cvt.MatLike] = list(x)  # equivalent to [i for i in x]
        tracks: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = self.yolo_tracker(x[-1])
        return np.asarray(im), tracks

    @override
    def get_imgs_and_tracks(self, x):
        imgs, tracks = self._tracker_infer_intent_batch(x)
        return imgs, tracks

    @override
    def generate_samples(self, imgs, tracks, x: T_intentBatch) -> list[T_intentSample]:
        ts = imgs.shape[0]
        samples: list[T_intentSample] = []
        for track_id, (
            boxes,
            poses,
            pose_masks,
            box_confs,
            pose_confs,
        ) in tracks.items():
            if (
                len(boxes) < self.args.observe_length
                or len(poses) < self.args.observe_length
            ):
                continue
            sample: T_intentSample = {  # type: ignore[reportAssignmentType]
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
                "bbox_conf": box_confs,
                "pose_conf": pose_confs,
            }
            samples.append(sample)

        return samples

    @override
    def pack_preds(self, x, preds):
        past_boxes = x["bboxes"].to(DEVICE)
        future_boxes = preds.to(DEVICE)
        past_future_boxes: torch.Tensor = torch.cat((past_boxes, future_boxes), dim=1)

        box_confs: torch.Tensor = x["bbox_conf"]
        pose_confs: torch.Tensor = x["pose_conf"]

        if self.return_dict:
            if self.return_pose:
                return PipelineResults(
                    bboxes=past_future_boxes,
                    bbox_conf=box_confs,
                    poses=x["pose"],
                    pose_conf=pose_confs,
                )
            return PipelineResults(bboxes=past_future_boxes, bbox_conf=box_confs)

        return preds


class HRNetTrackerWrapper(TrackerWrapper):
    def __init__(
        self,
        args: DefaultArguments,
        hrnet_m: str = "HRNet",
        hrnet_c: int = 48,
        hrnet_j: int = 17,
        hrnet_weights: str = "./weights/pose_hrnet_w48_384x288.pth",
        hrnet_joints_set: Literal["coco", "mpii"] = "coco",
        image_resolution: tuple[int, int] | str = (384, 288),
        single_person: bool = False,
        yolo_version: Literal["v3", "v5", "v8"] = "v8",
        use_tiny_yolo: bool = False,
        disable_tracking: bool = False,
        max_batch_size: int = 16,
        video_framerate: float = 30,
        device: torch.device | str | None = None,
        enable_tensorrt: bool = False,
    ) -> None:
        super().__init__()
        self.args = args
        if device is not None:
            device = torch.device(device)
        else:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        # WARNING: Yikes!
        if isinstance(image_resolution, str):
            image_resolution: tuple[int, int] = ast.literal_eval(image_resolution)

        match yolo_version:
            case "v3":
                if use_tiny_yolo:
                    yolo_model_def = "./models_/detectors/yolo/config/yolov3-tiny.cfg"
                    yolo_weights_path = (
                        "./models_/detectors/yolo/weights/yolov3-tiny.weights"
                    )
                else:
                    yolo_model_def = "./models_/detectors/yolo/config/yolov3.cfg"
                    yolo_weights_path = (
                        "./models_/detectors/yolo/weights/yolov3.weights"
                    )
                yolo_class_path = "./models_/detectors/yolo/data/coco.names"
            case "v5":
                if use_tiny_yolo:
                    yolo_model_def = "yolov5n"
                else:
                    yolo_model_def = "yolov5m"
                if enable_tensorrt:
                    yolo_trt_filename = f"{yolo_model_def}.engine"
                    if os.path.exists(os.path.normpath(yolo_trt_filename)):
                        yolo_model_def = yolo_trt_filename
                yolo_class_path = ""
                yolo_weights_path = ""
            case "v8":
                if use_tiny_yolo:
                    yolo_model_def = "yolov8s"
                else:
                    yolo_model_def = "yolov8m"
                yolo_class_path = ""
                yolo_weights_path = ""

        self.hrnet = SimpleHRNet(
            hrnet_c,
            hrnet_j,
            hrnet_weights,
            resolution=image_resolution,
            multiperson=not single_person,
            return_bounding_boxes=not disable_tracking,
            max_batch_size=max_batch_size,
            yolo_version=yolo_version,
            yolo_model_def=yolo_model_def,
            yolo_class_path=yolo_class_path,
            yolo_weights_path=yolo_weights_path,
            device=device,
            enable_tensorrt=enable_tensorrt,
        )
        self.disable_tracking = disable_tracking
        self.hrnet_joints_set = hrnet_joints_set

        self.cache: dict[
            int,
            dict[
                {
                    "bboxes": deque[npt.NDArray[np.float_]],
                    "pts": deque[npt.NDArray[np.float_]],
                }
            ],
        ] = {}

        self._prev_boxes: npt.NDArray[np.float_] | None = None
        self._prev_pts: npt.NDArray[np.float_] | None = None
        self._prev_person_ids: npt.NDArray[np.int_] | None = None
        self._next_person_id: int = 0
        self._frame_count: int = 0

    @override
    def __call__(
        self, image: torch.Tensor | npt.NDArray[np.uint8] | cvt.MatLike
    ) -> dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        self._frame_count += 1

        if isinstance(image, torch.Tensor):
            image = image.squeeze()
            if image.ndim == 4:
                image = image[-1]
            assert image.ndim == 3, "Input tensor must be of shape (C, H, W)."
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255.0).astype(np.uint8)
        else:
            assert image.ndim == 3, "Input must be of shape (H, W, C)."

        pts = self.hrnet.predict(image)

        if not self.disable_tracking:
            boxes, original_boxes, pts = pts
            if len(pts) > 0:
                if (
                    self._prev_boxes is None
                    and self._prev_pts is None
                    and self._prev_person_ids is None
                ):
                    person_ids = np.arange(
                        self._next_person_id,
                        len(pts) + self._next_person_id,
                        dtype=np.int_,
                    )
                    self._next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes,
                        pts=pts,
                        prev_boxes=self._prev_boxes,  # type: ignore[reportArgumentType]
                        prev_pts=self._prev_pts,  # type: ignore[reportArgumentType]
                        prev_person_ids=self._prev_person_ids,  # type: ignore[reportArgumentType]
                        next_person_id=self._next_person_id,
                        pose_alpha=0.2,
                        similarity_threshold=0.4,
                        smoothing_alpha=0.1,
                    )
                    assert (
                        pts.shape[-1] == 3 and pts.shape[-2] == 17
                    ), f"Invalid shape {pts.shape} for keypoints."
                    self._next_person_id = max(
                        self._next_person_id, np.max(person_ids) + 1
                    )
            else:
                person_ids = np.array((), dtype=np.int_)

            self._prev_boxes = boxes.copy()
            self._prev_pts = pts.copy()
            self._prev_person_ids = person_ids
        else:
            person_ids = np.arange(len(pts), dtype=np.int_)

        iterable = (
            zip(pts, person_ids)
            if self.disable_tracking
            else zip(original_boxes, pts, person_ids)
        )

        # person_id: (bboxes, poses, pose masks, bbox conf)
        ret: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

        for unpackable_values in iterable:
            if self.disable_tracking:
                pt, person_id = unpackable_values
                if (entry := self.cache.get(person_id, None)) is not None:
                    entry["pts"].append(pt)
                else:
                    self.cache[person_id] = {
                        "bboxes": deque([], maxlen=self.args.observe_length),
                        "pts": deque([pt], maxlen=self.args.observe_length),
                    }
            else:
                box, pt, person_id = unpackable_values
                assert pt.ndim == 2, f"Invalid shape {pt.shape} for keypoints."
                assert pt.shape == (17, 3), f"Invalid shape {pt.shape} for keypoints."

                if (entry := self.cache.get(person_id, None)) is not None:
                    entry["bboxes"].append(box)
                    entry["pts"].append(pt)
                    assert (
                        pt.shape[-1] == 3
                    ), f"Last dimension of keypoints must be 3, but is of shape {pt.shape} instead."

                else:
                    self.cache[person_id] = {
                        "bboxes": deque([box], maxlen=self.args.observe_length),
                        "pts": deque([pt], maxlen=self.args.observe_length),
                    }

        for person_id, entry in self.cache.items():
            if person_id not in person_ids:
                continue
            pts = np.array(entry["pts"])
            assert pts.ndim == 3, f"Invalid shape {pts.shape} for keypoints."
            assert pts.shape[1:] == (17, 3), f"Invalid shape {pts.shape} for keypoints."

            bboxes = np.array(entry["bboxes"], dtype=np.float_)
            boxes = bboxes[:, :-1]
            box_conf = bboxes[:, -1]

            # NOTE: This creates the bboxes from the pose keypoints as the HRNet model
            # mangles the bounding boxes when tracking is enabled.
            # boxes = np.zeros((pts.shape[0], 4), dtype=np.float_)
            # boxes[:, 0] = pts[:, :, 0].min(axis=1)
            # boxes[:, 1] = pts[:, :, 1].min(axis=1)
            # boxes[:, 2] = pts[:, :, 0].max(axis=1)
            # boxes[:, 3] = pts[:, :, 1].max(axis=1)

            boxes[..., 0::2] = boxes[..., 0::2] / image.shape[0]
            boxes[..., 1::2] = boxes[..., 1::2] / image.shape[1]

            boxes = np.clip(boxes, 0, 1)

            pts_conf: torch.Tensor = (
                torch.from_numpy(pts[:, :, 2].copy()).type(FloatTensor).to(DEVICE)
            )
            pts_mask = (pts_conf == 0.0).bool()

            ret[person_id] = (
                torch.from_numpy(boxes).type(FloatTensor).to(DEVICE),
                torch.from_numpy(pts[..., :2][..., ::-1].copy())
                .type(FloatTensor)
                .to(DEVICE),
                pts_mask,
                torch.from_numpy(box_conf).type(FloatTensor).to(DEVICE),
                pts_conf,
            )

        return ret

    @override
    def reset_tracker(self) -> None:
        self._frame_count = 0
        self._prev_boxes = None
        self._prev_pts = None
        self._prev_person_ids = None
        self._next_person_id = 0


class HRNetPipelineModelWrapper(PipelineWrapper):
    def __init__(
        self,
        args: DefaultArguments,
        model: nn.Module,
        tracker: HRNetTrackerWrapper,
        return_dict: bool = False,
        return_pose: bool = False,
    ) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.tracker: HRNetTrackerWrapper = tracker
        self.return_dict = return_dict
        self.return_pose = return_pose

    def _tracker_infer_intent_batch(self, x: T_intentBatch) -> tuple[
        torch.Tensor,
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        imgs = x["image"]
        assert isinstance(imgs, torch.Tensor), "input images must not be an empty list"
        imgs = imgs.squeeze()
        assert (
            imgs.dim() == 4
        ), f"input dimensionality must be 4, but is of shape {imgs.shape}"
        pts = self.tracker(imgs[-1].unsqueeze(0))
        return imgs, pts

    def _tracker_infer_intent_ndarray(self, x: npt.NDArray[np.uint8]) -> tuple[
        npt.NDArray[np.uint8],
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        tracks = self.tracker(x[-1])
        return x, tracks

    def _tracker_infer_intent_deque(self, x: deque[cvt.MatLike]) -> tuple[
        npt.NDArray[np.uint8],
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        im: list[cvt.MatLike] = list(x)
        tracks = self.tracker(x[-1])
        return np.asarray(im), tracks

    @override
    def get_imgs_and_tracks(self, x):
        imgs, tracks = self._tracker_infer_intent_batch(x)
        return imgs, tracks

    @override
    def generate_samples(self, imgs, tracks, x):
        ts = imgs.shape[0]
        samples: list[T_intentSample] = []

        for track_id, (
            boxes,
            poses,
            pose_masks,
            box_confs,
            pose_confs,
        ) in tracks.items():
            if (
                len(boxes) < self.args.observe_length
                or len(poses) < self.args.observe_length
            ):
                continue

            print(poses.min(), poses.max(), poses.mean(), poses.std())

            # NOTE: This scales up the poses in a hardcoded manner.
            scaled_poses = deepcopy(poses).type(FloatTensor)

            scaled_poses[..., 0] = scaled_poses[..., 0] / 640.0 * 1280.0
            scaled_poses[..., 1] = scaled_poses[..., 1] / 640.0 * 720.0

            sample: T_intentSample = {  # type: ignore[reportAssignmentType]
                "image": imgs,
                "pose": scaled_poses,
                "bboxes": boxes,
                "frames": x["frames"].squeeze(),
                "ped_id": [f"track_{track_id:03d}"] * ts,
                "video_id": x["video_id"],
                "total_frames": x["total_frames"].squeeze(),
                "pose_masks": pose_masks,
                "local_featmaps": x["local_featmaps"],
                "intention_prob": np.zeros((ts), dtype=np.float_),
                "intention_binary": np.zeros((ts), dtype=np.int_),
                "disagree_score": np.zeros((ts), dtype=np.float_),
                "global_featmaps": x["global_featmaps"],
                "original_bboxes": boxes,
                "bbox_conf": box_confs,
                "pose_conf": pose_confs,
            }
            samples.append(sample)

        return samples

    @override
    def pack_preds(self, x, preds):
        past_boxes = x["original_bboxes"].to(DEVICE)
        future_boxes = preds.to(DEVICE)

        box_confs: torch.Tensor = x["bbox_conf"]
        pose_confs: torch.Tensor = x["pose_conf"]

        if self.return_dict:
            res = PipelineResults(past_boxes, future_boxes, box_confs)
            if self.return_pose:
                res.poses = x["pose"]
                res.pose_conf = pose_confs
            return res

        return preds


class GTTrackingIntentPipelineWrapper(PipelineWrapper):
    def __init__(
        self,
        args: DefaultArguments,
        model: nn.Module,
        db: T_intentDB,
        return_dict: bool = False,
        normalize_bbox: bool = False,
    ):
        super().__init__()
        self.args = args
        self.model = model
        self.db = db
        self.return_dict = return_dict
        self.normalize_bbox = normalize_bbox

    def __getitem__(self, video_id: str, frames: Sequence[int]) -> (
        dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]
        | None
    ):
        working_dict: dict[
            str,
            dict[
                {
                    "bbox": list[tuple[float, float, float, float]],
                    "bbox_conf": list[float],
                    "pose": list[list[tuple[float, float]]],
                    "pose_mask": list[list[bool]],
                    "pose_conf": list[list[float]],
                }
            ],
        ] = {}

        for ped_id, annotations in self.db[video_id].items():
            # extract frames from annotations within `frames`.
            if ped_id not in working_dict:
                working_dict[ped_id] = {
                    "bbox": [],
                    "bbox_conf": [],
                    "pose": [],
                    "pose_mask": [],
                    "pose_conf": [],
                }

            cv_anns = annotations["cv_annotations"]

            for ann_frame in annotations["frames"]:
                if ann_frame in frames:
                    ann_subidx = annotations["frames"].index(ann_frame)
                    ### Bounding Boxes ###
                    bbox = cv_anns["bbox"][ann_subidx]
                    # GUARD
                    assert all(
                        isinstance(x, float) for x in bbox
                    ), f"bbox coordinates not of type float: {bbox}"
                    if (bbox_list := working_dict[ped_id].get("bbox")) is not None:
                        bbox_list.append(tuple(bbox))
                        working_dict[ped_id]["bbox_conf"].append(1.0)
                    else:
                        working_dict[ped_id]["bbox"] = [tuple(bbox)]
                        working_dict[ped_id]["bbox_conf"] = [1.0]
                    ### Poses ###
                    pose = cv_anns["skeleton"][ann_subidx]
                    pose_mask = cv_anns["observed_skeleton"][ann_subidx]

                    assert len(pose) == 17, f"pose length not 17! {pose}"
                    if (pose_list := working_dict[ped_id].get("pose")) is not None:
                        pose_list.append(pose)
                        working_dict[ped_id]["pose_mask"].append(pose_mask)
                        working_dict[ped_id]["pose_conf"].append(
                            [float(x) for x in pose_mask]
                        )
                    else:
                        working_dict[ped_id]["pose"] = [pose]
                        working_dict[ped_id]["pose_mask"] = [pose_mask]
                        working_dict[ped_id]["pose_conf"] = [
                            [float(x) for x in pose_mask]
                        ]

        # Convert format
        res: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        for track_id, (ped_id, annotations) in enumerate(working_dict.items()):
            try:
                if len(annotations["bbox"]) < self.args.observe_length:
                    continue
            except KeyError:
                print(annotations)
                continue

            bbox = torch.from_numpy(np.asarray(annotations["bbox"], dtype=np.float_))
            bbox_conf = torch.from_numpy(
                np.asarray(annotations["bbox_conf"], dtype=np.float_)
            )
            pose = torch.from_numpy(np.asarray(annotations["pose"], dtype=np.float_))
            pose_conf = torch.from_numpy(
                np.asarray(annotations["pose_conf"], dtype=np.float_)
            )
            pose_mask = torch.from_numpy(
                np.asarray(annotations["pose_mask"], dtype=np.float_)
            )

            res[track_id] = (bbox, pose, pose_mask, bbox_conf, pose_conf)

        if len(res.keys()) == 0:
            return None

        return res

    @override
    def get_imgs_and_tracks(self, x):
        imgs = x["image"]
        assert not isinstance(imgs, list), "imgs should not be an empty list"

        video_id: str = x["video_id"][0][0]
        frames: list[int] = x["frames"]

        tracks = self.__getitem__(video_id, frames)

        if tracks is None:
            return imgs, {
                0: (
                    torch.Tensor(),
                    torch.Tensor(),
                    torch.Tensor(),
                    torch.Tensor(),
                    torch.Tensor(),
                )
            }

        return imgs, tracks

    @override
    def generate_samples(self, imgs, tracks, x):
        ts = imgs.shape[0]
        samples: list[T_intentSample] = []

        for track_id, (
            boxes,
            poses,
            pose_masks,
            box_confs,
            pose_confs,
        ) in tracks.items():
            if any(
                len(x) < self.args.observe_length
                for x in [boxes, poses, box_confs, pose_confs]
            ):
                continue

            # NOTE: Since we know that the boxes are not normalized from the ground
            # truth annotations, we can just scale them down here as the visualisation
            # pipeline expects non-normalized boxes.
            og_bboxes = deepcopy(boxes)
            og_bboxes[..., 0::2] = og_bboxes[..., 0::2] / self.args.image_shape[0]
            og_bboxes[..., 1::2] = og_bboxes[..., 1::2] / self.args.image_shape[1]

            sample: T_intentSample = {  # type: ignore[reportAssignmentType]
                "image": imgs,
                "pose": poses,
                "bboxes": og_bboxes if self.normalize_bbox else boxes,
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
                "original_bboxes": og_bboxes,
                "bbox_conf": box_confs,
                "pose_conf": pose_confs,
            }
            samples.append(sample)

        return samples

    @override
    def pack_preds(self, x, preds):
        if self.args.task_name == "ped_intent":
            preds = F.sigmoid(preds)

            if preds.ndim == 0:
                preds = preds.unsqueeze(0)

            if self.return_dict:
                bbox = x["original_bboxes"]
                return PipelineResults(
                    bboxes=bbox,
                    intent=preds,
                    bbox_conf=x["bbox_conf"],
                    poses=x["pose"],
                    pose_conf=x["pose_conf"],
                )

        elif self.args.task_name == "ped_traj":
            past_boxes = x["original_bboxes"].to(DEVICE)
            future_boxes = preds.to(DEVICE)

            box_confs: torch.Tensor = x["bbox_conf"]
            pose_confs: torch.Tensor = x["pose_conf"]

            if self.return_dict:
                res = PipelineResults(
                    past_boxes,
                    future_boxes,
                    box_confs,
                    poses=x["pose"],
                    pose_conf=pose_confs,
                )
                return res
        else:
            raise NotImplementedError("Driving decision not implemented")

        return preds


def main(args: DefaultArguments):
    args.batch_size = 1
    args.database_file = "traj_database_train.pkl"
    args.intent_model = False
    args.traj_model = True
    args.traj_loss = ["nll"]
    args.model_name = "transformer_bbox_traj"
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

    yolo_tracker = YOLOTrackerWrapper(args, yolo_model, tracker)
    pipeline_model = YOLOPipelineModelWrapper(args, model, yolo_tracker)

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
