from __future__ import annotations

import colorsys
import hashlib
import os
import traceback
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot import BoTSORT, BYTETracker, DeepOCSORT
from boxmot.trackers.basetracker import BaseTracker
from cv2 import typing as cvt
from numpy import typing as npt
from torch import nn
from torch.utils.data import default_collate
from torchvision.transforms import v2
from tqdm.auto import tqdm
from typing_extensions import override
from ultralytics import YOLO
from ultralytics.engine.results import Results

from data.custom_dataset import T_intentBatch, T_intentSample
from eval import load_args, load_model_state_dict
from models.build_model import build_model
from models.pose_pipeline.pose import (
    HRNetPipelineModelWrapper,
    HRNetTrackerWrapper,
    PipelineResults,
    PipelineWrapper,
    TrackerWrapper,
    YOLOPipelineModelWrapper,
    YOLOTrackerWrapper,
)
from opts import get_opts, init_args
from utils.args import DefaultArguments
from utils.cuda import *
from utils.plotting import PipelinePlotter, PosePlotter

SAVE_VIDEO = False


def plot_box_on_img(
    img: npt.NDArray[np.uint8] | cvt.MatLike,
    box: tuple[float, float, float, float],
    conf: float,
    cls: int,
    id: int,
) -> npt.NDArray[np.uint8] | cvt.MatLike:
    """Draws a bounding box with ID, confidence, and class information

    :param img: The image array to draw on.
    :type img: npt.NDArray[np.uint8] or cvt.MatLike
    :param box: The bounding box coordinates as (x1, y1, x2, y2).
    :type box: tuple[float, float, float, float]
    :param float conf: Confidence score of the detection.
    :param int cls: Class ID of the detection.
    :param int id: Unique identifier for the detection.
    :return: The image array with the bounding box drawn on it.
    :rtype: npt.NDArray[np.uint8] or cvt.MatLike
    """
    thickness = 2
    fontscale = 0.5

    img = cv2.rectangle(
        img,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        id_to_color(id),
        thickness,
    )

    img = cv2.putText(
        img,
        f"id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}",
        (int(box[0]), int(box[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontscale,
        id_to_color(id),
        thickness,
    )

    return img


def id_to_color(
    id: int, saturation: float = 0.75, value: float = 0.95
) -> tuple[int, int, int]:
    """Generates a consistent unique BGR color for a given ID using hashing.

    :param int id: Unique identifier for which to generate a color.
    :param float saturation: Saturation value for the color in HSV space, defaults to
    0.75.
    :param float value: Value (brightness) for the color in HSV space, defaults to 0.95.
    :return: A tuple representing the BGR color.
    :rtype: tuple[int, int, int]
    """

    # Hash the ID to get a consistent unique value
    hash_object = hashlib.sha256(str(id).encode())
    hash_digest = hash_object.hexdigest()

    # Convert the first few characters of the hash to an integer
    # and map it to a value between 0 and 1 for the hue
    hue = int(hash_digest[:8], 16) / 0xFFFFFFFF

    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
    rgb_255 = tuple(int(component * 255) for component in rgb)
    hex_color = "#%02x%02x%02x" % rgb_255
    # Strip the '#' character and convert the string to RGB integers
    rgb = tuple(int(hex_color.strip("#")[i : i + 2], 16) for i in (0, 2, 4))

    # Convert RGB to BGR for OpenCV
    bgr = rgb[::-1]

    assert len(bgr) == 3, f"Invalid BGR color: {bgr}"

    return bgr


def plot_past_future_traj(
    args: DefaultArguments,
    im: npt.NDArray[np.uint8] | cvt.MatLike,
    boxes: list[npt.NDArray[np.float32]] | npt.NDArray[np.float32],
    track_id: int,
) -> cvt.MatLike | npt.NDArray[np.uint8]:
    """Plots past and future trajectories of all detected pedestrians.

    :param DefaultArguments args: Default arguments for training.
    :param im: Current frame.
    :type im: npt.NDArray[np.uint8] or cvt.MatLike
    :param npt.NDArray[np.float32] boxes: Bounding boxes of detected pedestrians that
    are already tracked. It is in the format [num_boxes, max_track_size, 4] where 4
    represents the bounding box coordinates (x1, y1, x2, y2), where the coordinates are
    non-normalized.
    :param BaseTracker tracker: The tracker object.
    :param int track_id: The unique identifier of the tracked object for color
    consistency.
    :return: Image with past and future trajectories of all detected pedestrians.
    :rtype: npt.NDArray[np.uint8] or cvt.MatLike
    """
    for i, box in enumerate(boxes):
        if i < args.observe_length:
            trajectory_thickness = int(np.sqrt(float(i + 1)) * 1.2)
        else:
            scalar = (
                args.observe_length - (i / args.predict_length) * args.observe_length
            )
            scalar = max(0.0, scalar)
            trajectory_thickness = int(np.sqrt(float(scalar)) * 1.2)
        centre = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        im = cv2.circle(
            im,
            centre,
            2,
            color=id_to_color(track_id),
            thickness=trajectory_thickness,
        )
    return im


def scale_bboxes_up(
    bboxes: npt.NDArray[np.float_] | torch.Tensor,
    scale: float | tuple[float, float] | tuple[int, int],
) -> npt.NDArray[np.float_]:
    """Scales up the bounding box coordinates by a given scale.

    :param bboxes: Normalised bounding box coordinates in the format [x1, y1, x2, y2] with
    values in (0, 1), shape [num_boxes, 4].
    :type bboxes: npt.NDArray[np.float_] or torch.Tensor
    :param scale: Scale factor to scale up the bounding box coordinates.
    :type scale: float or tuple[float, float] or tuple[int, int]
    :return: Scaled up bounding box coordinates.
    :rtype: npt.NDArray[np.float_]
    """

    scale_x: float
    scale_y: float

    match scale:
        case float(x):
            scale_x = scale_y = x
        case (x, y):
            scale_x = float(x)
            scale_y = float(y)
        case _:
            raise ValueError(f"Invalid scale type: `{type(scale)}`")

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()

    bboxes[:, 0::2] = bboxes[:, 0::2] * scale_x
    bboxes[:, 1::2] = bboxes[:, 1::2] * scale_y

    return bboxes


def unscale_then_rescale_poses(
    poses: npt.NDArray[np.float_], unscale: tuple[int, int], rescale: tuple[int, int]
) -> npt.NDArray[np.float_]:
    """Unscales the poses by a given scale and then rescales them by another scale.

    :param npt.NDArray[np.float_] poses: Pose keypoints in the format [num_boxes, 1, 17, 3].
    :param tuple[int, int] unscale: Scale factor to unscale the pose keypoints.
    :param tuple[int, int] rescale: Scale factor to rescale the pose keypoints.
    :return: Rescaled pose keypoints.
    :rtype: npt.NDArray[np.float_]
    """

    unscale_x, unscale_y = unscale
    rescale_x, rescale_y = rescale

    poses[..., 0] = poses[..., 0] / float(unscale_x)
    poses[..., 1] = poses[..., 1] / float(unscale_y)

    poses[..., 0] = poses[..., 0] * float(rescale_x)
    poses[..., 1] = poses[..., 1] * float(rescale_y)

    return poses


class DirtyGotoException(Exception):
    pass


def infer(
    args: DefaultArguments,
    pipeline_model: PipelineWrapper,
    source: str | Path,
    transform: v2.Compose | None,
    save_video: bool = SAVE_VIDEO,
    save_path: str | Path = "yolo_pipeline.avi",
):
    _ = pipeline_model.eval()

    source = source if isinstance(source, str) else str(source)
    vid = cv2.VideoCapture(source)
    frame_number = 0

    buffer: deque[cvt.MatLike | torch.Tensor] = deque(maxlen=args.observe_length)
    pose_plotter = PosePlotter()

    video_writer: cv2.VideoWriter | None = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    with tqdm(total=vid.get(cv2.CAP_PROP_FRAME_COUNT), leave=True) as pbar:
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
                if transform is not None:
                    buffer.append(transform(im))
                else:
                    buffer.append(im)

                # NOTE: This is a silly way to create a GOTO statement with the
                # `finally` clause of a try-except block.
                if len(buffer) < args.observe_length:
                    raise DirtyGotoException("Buffer is not full yet.")

                sample: T_intentSample = {
                    "image": torch.stack(tuple(buffer)).to(DEVICE),
                    "bboxes": np.zeros((1, args.observe_length, 4), dtype=np.float32),
                    "pose": np.zeros((1, args.observe_length, 17, 3), dtype=np.float32),
                    "intent": np.zeros((1, args.observe_length, 2), dtype=np.float32),
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
                    "video_id": [source.split("_")[1]] * args.max_track_size,
                }

                batch_tensor: T_intentBatch = default_collate([sample])

                results: PipelineResults = pipeline_model(batch_tensor)
                _ = buffer.popleft()

                assert (
                    results.bboxes.ndim == 3
                ), f"Bounding boxes must be of shape (n_dets, 60, 4) but is of shape {results.bboxes.shape} instead."
                if torch.all(results.bboxes == 0.0):
                    raise DirtyGotoException("No bounding boxes detected.")

                assert (
                    results.bbox_conf is not None
                ), "Confidence levels are not detected."

                assert results.poses is not None, "Poses are not detected."
                assert results.poses.ndim == 4 and results.poses.shape[1:] == (
                    15,
                    17,
                    2,
                ), f"Poses must be of shape (n_dets, 15, 17, 2) but is of shape {results.poses.shape} instead."

                assert results.pose_conf.ndim == 3 and results.pose_conf.shape[1:] == (
                    15,
                    17,
                ), f"Pose confidence must be of shape (n_dets, 15, 17) but is of shape {results.pose_conf.shape} instead."

                for track_id, (bboxes, bbox_conf, poses, pose_conf) in enumerate(
                    zip(
                        results.bboxes,
                        results.bbox_conf,
                        results.poses,
                        results.pose_conf,
                    )
                ):
                    rescaled_bboxes = scale_bboxes_up(
                        bboxes.detach().cpu().numpy(), args.image_shape
                    )
                    if torch.count_nonzero(bboxes) > 15 * 4:
                        bbox = rescaled_bboxes[args.observe_length - 1, :].reshape((4))
                        im = plot_box_on_img(
                            im, bbox, bbox_conf.mean().item(), 0, track_id
                        )
                        future_bbox = rescaled_bboxes[-1, :].reshape((4))
                        im = plot_box_on_img(
                            im, future_bbox, bbox_conf.mean().item(), 0, track_id
                        )
                    im = plot_past_future_traj(
                        args,
                        im,
                        rescaled_bboxes,
                        track_id,
                    )
                    last_pose = poses[-1, :, :].detach().cpu().numpy()
                    last_pose_conf = np.expand_dims(
                        pose_conf[-1, :].detach().cpu().numpy(), axis=1
                    )
                    last_combined_pose = np.concatenate(
                        (last_pose, last_pose_conf), axis=1
                    )
                    last_combined_pose = unscale_then_rescale_poses(
                        last_combined_pose, (640, 640), args.image_shape
                    )
                    im = PipelinePlotter.draw_points_and_skeleton(
                        im,
                        last_combined_pose,
                        "coco",
                        person_index=track_id,
                        points_color_palette="gist_rainbow",
                        skeleton_color_palette="jet",
                        points_palette_samples=10,
                        flip_xy=True,
                    )
                # last_poses = (
                #     results.poses[:, -1, :, :].unsqueeze(1).detach().cpu().numpy()
                # )
                # last_confs = (
                #     results.pose_conf[:, -1, :]
                #     .unsqueeze(1)
                #     .detach()
                #     .cpu()
                #     .numpy()
                #     .reshape(-1, 1, 17, 1)
                # )
                # # extend last dim to 3 for plotting
                # last_poses = np.concatenate(
                #     [last_poses, last_confs],
                #     axis=3,
                # )
                # last_poses = unscale_then_rescale_poses(
                #     last_poses, (640, 640), args.image_shape
                # )
                # pose_plotter.plot_keypoints(im, last_poses)
            except DirtyGotoException:
                continue
            except Exception as e:
                pbar.write(f"Error: {e}, {frame_number:04d}, {traceback.format_exc()}")
                break
            finally:
                cv2.imshow("YOLO Tracker", im)

                if save_video and video_writer:
                    video_writer.write(im)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") or key == ord("q"):
                    break

    if save_video and video_writer:
        video_writer.release()

    vid.release()
    cv2.destroyAllWindows()


def main(args: DefaultArguments):
    source = args.source
    yolo_pipeline_weights = args.yolo_pipeline_weights
    boxmot_tracker_weights = args.boxmot_tracker_weights
    tracker_type = args.tracker
    hrnet_yolo_ver = args.hrnet_yolo_ver
    save_output: bool = args.save_output
    output: str = args.output

    args.batch_size = 1
    args.intent_model = False
    args.traj_model = True
    args.traj_loss = ["bbox_l2"]
    args.model_name = "tcan_traj_bbox"
    # args.model_name = "transformer_traj_bbox"
    args.loss_weights = {
        "loss_intent": 0.0,
        "loss_traj": 1.0,
        "loss_driving": 0.0,
    }
    args.load_image = True
    args.backbone = ""
    args.freeze_backbone = False
    args.observe_length = 15
    args.predict_length = 45
    args.max_track_size = args.observe_length + args.predict_length
    args.kernel_size = 2
    args.n_layers = 4

    if "transformer" in args.model_name:
        args.observe_length += 1
        args.max_track_size += 1

    model, _, _ = build_model(args)
    if args.compile_model:
        model = torch.compile(
            model, options={"triton.cudagraphs": True}, fullgraph=True
        )
    model = nn.DataParallel(model)

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
        pipeline_wrapper = YOLOPipelineModelWrapper(
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
        pipeline_wrapper = HRNetPipelineModelWrapper(
            args,
            model,
            tracker_wrapper,
            return_dict=True,
            return_pose=True,
        )
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

    args = parser.parse_args()

    main(args)
