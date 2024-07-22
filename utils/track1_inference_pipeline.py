from __future__ import annotations

import os
import pickle
import traceback
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from boxmot import BoTSORT, BYTETracker, DeepOCSORT
from cv2 import typing as cvt
from numpy import typing as npt
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from torchvision.transforms import v2
from tqdm.auto import tqdm
from typing_extensions import override
from ultralytics import YOLO

from data.custom_dataset import T_intentBatch, T_intentSample
from database.create_database import T_intentDB
from eval import load_args, load_model_state_dict
from models.build_model import build_model
from models.pose_pipeline.pose import (
    BaseTracker,
    GTTrackingIntentPipelineWrapper,
    HRNetPipelineModelWrapper,
    HRNetTrackerWrapper,
    PipelineResults,
    PipelineWrapper,
    TrackerWrapper,
    YOLOPipelineModelWrapper,
    YOLOTrackerWrapper,
)
from opts import init_args
from utils.args import DefaultArguments
from utils.cuda import *
from utils.track2_inference_pipeline import (
    DirtyGotoException,
    id_to_color,
    plot_box_on_img,
    plot_past_future_traj,
    scale_bboxes_up,
)


def plot_box_intent_on_img(
    img: npt.NDArray[np.uint8] | cvt.MatLike,
    box: tuple[float, float, float, float],
    conf: float,
    intent: float,
    cls: int,
    id: int,
) -> npt.NDArray[np.uint8] | cvt.MatLike:
    """Plot the box on the image with the crossing intent.

    :param img: The image to plot the box on.
    :type img: npt.NDArray[np.uint8] or cvt.MatLike
    :param box: The box to plot.
    :type box: tuple[float, float, float, float]
    :param float conf: The confidence for the bounding box.
    :param float intent: The intent to cross the road.
    :param int cls: The class of the bounding box.
    :param int id: The id of the bounding box.
    """
    thickness = 2
    fontscale = 0.5

    # Plot the box
    im = plot_box_on_img(img, box, conf, cls, id)

    # Draw the crossing confidence
    crossing_conf = abs(intent - 0.5) * 2
    color = id_to_color(id)
    im = cv2.putText(
        im,
        f"intent_conf: {crossing_conf:.2f}",
        (int(box[2]), int(box[1]) + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontscale,
        color,
        thickness,
    )

    # Convert to PIL image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(im)

    # Draw the intent
    font = ImageFont.truetype("seguisym.ttf", size=30)
    draw = ImageDraw.Draw(pil_im, "RGBA")
    text = "🚶" if intent > 0.5 else "🚹"
    color = (*color, int(crossing_conf * 255))
    draw.text((int(box[2]), int(box[1]) - 10), text, font=font, fill=color)

    # Return the image
    im = np.asarray(pil_im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im


@dataclass
class IntentPipelineResults:
    bboxes: torch.Tensor
    intent: torch.Tensor
    bbox_conf: torch.Tensor | npt.NDArray[np.float_] | None


class YOLOIntentPipelineWrapper(YOLOPipelineModelWrapper):
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

            # NOTE: Since we know that the boxes are normalized from the HRNet tracker
            # pipeline, we can just scale them up here as the inferencing models were
            # trained on non-normalized boxes.
            scaled_boxes = deepcopy(boxes)
            scaled_boxes[..., 0::2] = boxes[..., 0::2] * self.args.image_shape[0]
            scaled_boxes[..., 1::2] = boxes[..., 1::2] * self.args.image_shape[1]

            sample: T_intentSample = {  # type: ignore[reportAssignmentType]
                "image": imgs,
                "pose": poses,
                "bboxes": scaled_boxes,
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
        preds = F.sigmoid(preds)

        if preds.ndim == 0:
            preds = preds.unsqueeze(0)

        if self.return_dict:
            return PipelineResults(
                bboxes=x["original_bboxes"], bbox_conf=x["bbox_conf"], intent=preds
            )
        return preds


class HRNetIntentPipelineWrapper(HRNetPipelineModelWrapper):
    def __init__(
        self,
        args: DefaultArguments,
        model: nn.Module,
        tracker: HRNetTrackerWrapper,
        return_dict: bool = True,
    ):
        super().__init__(args, model, tracker, return_dict=return_dict)

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

            # NOTE: Since we know that the boxes are normalized from the HRNet tracker
            # pipeline, we can just scale them up here as the inferencing models were
            # trained on non-normalized boxes.
            original_boxes = boxes.clone()
            boxes[..., 0::2] = boxes[..., 0::2] * self.args.image_shape[0]
            boxes[..., 1::2] = boxes[..., 1::2] * self.args.image_shape[1]

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
                "original_bboxes": original_boxes,
                "bbox_conf": box_confs,
                "pose_conf": pose_confs,
            }
            samples.append(sample)

        return samples

    @override
    def pack_preds(self, x, preds):
        preds = F.sigmoid(preds)

        if preds.ndim == 0:
            preds = preds.unsqueeze(0)

        if self.return_dict:
            return PipelineResults(
                bboxes=x["original_bboxes"],
                bbox_conf=x["bbox_conf"],
                intent=preds,
            )

        return preds


def infer(
    args: DefaultArguments,
    pipeline_model: PipelineWrapper,
    source: str | Path,
    transform: v2.Compose | None = None,
    save_video: bool = False,
    save_path: str | Path = "yolo_pipeline.avi",
):
    _ = pipeline_model.eval()

    source = source if isinstance(source, str) else str(source)
    vid = cv2.VideoCapture(source)
    frame_number = 0

    buffer: deque[cvt.MatLike | torch.Tensor] = deque(maxlen=args.observe_length)

    video_writer: cv2.VideoWriter | None = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video_id = os.path.normpath(source).split(os.sep)[-1]
    video_id = os.path.splitext(video_id)[0]

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    time_elapsed: list[float] = []

    with tqdm(total=total_frames, leave=True) as pbar:
        while True:
            _ = pbar.update(1)
            ret, im = vid.read()

            if not ret:
                pbar.write(f"EOF at frame {frame_number:04d}")
                break

            if save_video and video_writer is None:
                video_writer = cv2.VideoWriter(
                    save_path,
                    fourcc,
                    30,
                    (im.shape[1], im.shape[0]),
                )

            try:
                frame_number += 1
                buffer.append(im if transform is None else transform(im))

                if len(buffer) < args.observe_length:
                    raise DirtyGotoException("Buffer is not full yet.")

                sample: T_intentSample = {  # type: ignore[reportAssignmentType]
                    "image": torch.stack(tuple(buffer)).to(DEVICE),
                    "bboxes": np.zeros((1, args.observe_length, 4), dtype=np.float32),
                    "pose": np.zeros((1, args.observe_length, 17, 3), dtype=np.float32),
                    "intent": np.zeros((1, args.observe_length, 1), dtype=np.float32),
                    "frames": np.array(
                        list(range(frame_number - args.observe_length, frame_number))
                    ),
                    "total_frames": np.array(
                        list(
                            range(
                                frame_number - args.observe_length,
                                frame_number + args.predict_length,
                            )
                        )
                    ),
                    "track_id": np.zeros((1, args.observe_length), dtype=np.int64),
                    "local_featmaps": [],
                    "global_featmaps": [],
                    "video_id": [video_id] * args.max_track_size,
                }

                batch_tensor: T_intentBatch = default_collate([sample])

                results: IntentPipelineResults | None = pipeline_model(batch_tensor)
                _ = buffer.popleft()

                if results is None:
                    raise DirtyGotoException("No results returned.")

                assert results.bbox_conf is not None
                assert results.intent is not None

                # NOTE: This is just to avoid iterating over a 0-d tensor, there should
                # be a better way.
                try:
                    for track_id, (bboxes, box_confs, intent) in enumerate(
                        zip(results.bboxes, results.bbox_conf, results.intent)
                    ):
                        rescaled_bboxes = scale_bboxes_up(
                            bboxes.detach().cpu().numpy(), args.image_shape
                        )

                        bbox = rescaled_bboxes[-1].reshape((4))
                        im = plot_box_intent_on_img(
                            im, bbox, box_confs[-1].item(), intent.item(), 0, track_id
                        )

                        im = plot_past_future_traj(args, im, rescaled_bboxes, track_id)
                except TypeError:
                    pbar.write(f"bboxes: {results.bboxes.shape}")
                    pbar.write(f"bbox_conf: {results.bbox_conf.shape}")
                    pbar.write(f"intent: {results.intent}")

                    raise DirtyGotoException

            except DirtyGotoException:
                continue
            except Exception as e:
                pbar.write(
                    f"Error {e} at frame {frame_number:04d}, {traceback.format_exc()}"
                )
                break
            finally:
                time_elapsed.append(pbar.format_dict["elapsed"] - sum(time_elapsed))
                cv2.imshow("Pedestrian Intent Prediction Pipeline", im)

                if save_video and video_writer:
                    video_writer.write(im)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") or key == ord("q"):
                    break

    if save_video and video_writer:
        video_writer.release()

    vid.release()
    cv2.destroyAllWindows()

    np_elapsed = np.array(time_elapsed[15:])
    print(
        f"Average time per frame: {np_elapsed.mean() * 1e3:0.3f}ms ± {np_elapsed.std() * 1e3:0.3f}ms"
    )


def main(args: DefaultArguments):
    source = args.source
    yolo_pipeline_weights = args.yolo_pipeline_weights
    boxmot_tracker_weights = args.boxmot_tracker_weights
    tracker_type = args.tracker
    hrnet_yolo_ver = args.hrnet_yolo_ver
    save_output: bool = args.save_output
    output: str = args.output
    load_db: bool = args.load_db

    args.batch_size = 1
    args.intent_model = True
    args.traj_model = False
    args.traj_loss = ["mse"]
    args.model_name = "tcn_int_bbox"
    args.loss_weights = {
        "loss_intent": 1.0,
        "loss_traj": 0.0,
        "loss_driving": 0.0,
    }
    args.load_image = True
    args.backbone = ""
    args.freeze_backbone = False
    args.observe_length = 15
    args.predict_length = 45
    args.max_track_size = args.observe_length + args.predict_length
    args.kernel_size = 8
    args.n_layers = 2

    model, _, _ = build_model(args)
    if args.compile_model:
        model = torch.compile(
            model, options={"triton.cudagraphs": True}, fullgraph=True
        )
    model = nn.DataParallel(model)

    db: T_intentDB | None = None
    if load_db:
        if not os.path.exists(
            db_path := os.path.join(
                args.dataset_root_path, args.database_path, args.database_file
            )
        ):
            raise FileNotFoundError(f"Database file not found at {db_path}")
        task = args.task_name.split("_")[1]
        with open(
            os.path.join(
                args.dataset_root_path, args.database_path, f"{task}_database_train.pkl"
            ),
            "rb",
        ) as f:
            imdb_train: T_intentDB = pickle.load(f)
        with open(
            os.path.join(
                args.dataset_root_path, args.database_path, f"{task}_database_val.pkl"
            ),
            "rb",
        ) as f:
            imdb_val: T_intentDB = pickle.load(f)
        with open(
            os.path.join(
                args.dataset_root_path, args.database_path, f"{task}_database_test.pkl"
            ),
            "rb",
        ) as f:
            imdb_test: T_intentDB = pickle.load(f)

        db = imdb_train
        for imdb in [imdb_val, imdb_test]:
            for vid in imdb:
                if vid not in db:
                    db[vid] = imdb[vid]

        del imdb_train, imdb_val, imdb_test

    if args.checkpoint_path != "ckpts":
        args = load_args(args.dataset_root_path, args.checkpoint_path)
        model = load_model_state_dict(args.checkpoint_path, model)

    tracker_wrapper: TrackerWrapper
    pipeline_wrapper: PipelineWrapper

    if tracker_type in ["botsort", "byte", "deepocsort"]:
        yolo_model = YOLO(yolo_pipeline_weights)
        tracker: BaseTracker
        match args.tracker:
            case "botsort":
                tracker = BoTSORT(
                    model_weights=Path(boxmot_tracker_weights),
                    device="cuda:0",
                    fp16=True,
                )
            case "byte":
                tracker = BYTETracker()
            case "deepocsort":
                tracker = DeepOCSORT(
                    model_weights=Path(boxmot_tracker_weights),
                    device="cuda:0",
                    fp16=True,
                )
            case _:
                raise ValueError(
                    f"Invalid tracker: {tracker_type}. This really shouldn't be possible."
                )
        tracker_wrapper = YOLOTrackerWrapper(
            args, yolo_model, tracker, persistent_tracker=True
        )
        pipeline_wrapper = YOLOIntentPipelineWrapper(
            args,
            model,
            tracker_wrapper,
            return_dict=True,
            return_pose=True,
            persistent_tracker=True,
        )
    elif tracker_type == "hrnet":
        tracker_wrapper = HRNetTrackerWrapper(
            args, yolo_version=hrnet_yolo_ver, device=DEVICE
        )
        pipeline_wrapper = HRNetIntentPipelineWrapper(
            args,
            model,
            tracker_wrapper,
            return_dict=True,
        )
    elif tracker_type == "demo":
        assert db is not None, f"Did not load db, thus gt annotations cannot be loaded"
        pipeline_wrapper = GTTrackingIntentPipelineWrapper(args, model, db, True)
    else:
        raise ValueError(f"Invalid tracker type: {tracker_type}")

    transform = v2.Compose(
        [
            v2.ToPILImage(),
            v2.Resize((640, 640)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    infer(
        args,
        pipeline_wrapper,
        source,
        transform,
        save_video=save_output,
        save_path=output,
    )


if __name__ == "__main__":
    # args = get_opts()
    parser = init_args()

    _ = parser.add_argument("--save_output", "-so", action="store_true")

    _ = parser.add_argument("--output", "-o", type=str, default="pipeline.avi")

    _ = parser.add_argument("--load_db", "-l", action="store_true")

    args = parser.parse_args()

    main(args)
