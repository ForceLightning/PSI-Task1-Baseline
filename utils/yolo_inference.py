from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from boxmot import BoTSORT, BYTETracker, DeepOCSORT
from numpy import typing as npt
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from opts import get_opts
from utils.args import DefaultArguments
from utils.cuda import *
from utils.plotting import PosePlotter


class DetectionResult(NamedTuple):
    coords: list[float]
    conf: float
    det_class: int


def plot_pose_from_video(
    video_path: str,
    yolo_model: YOLO,
    tracker: BoTSORT | BYTETracker | DeepOCSORT | None,
):
    if tracker is None:
        tracker = BoTSORT(
            model_weights=Path("osnet_x0_25_msmt17.pt"), device="cuda:0", fp16=True
        )
    pose_plotter = PosePlotter()

    vid = cv2.VideoCapture(video_path)
    frame_number = 0

    with tqdm(total=vid.get(cv2.CAP_PROP_FRAME_COUNT), position=0, leave=True) as pbar:
        while True:
            _ = pbar.update(1)
            ret, im = vid.read()
            if not ret:
                pbar.write("EOF or video not found.")
                break

            # Increment frame number
            frame_number += 1
            result: Results = yolo_model.predict(
                im, verbose=False, classes=[0], conf=0.15
            )
            dets: list[list[float | int]] = []
            kp_dets: list[list[float]] = []
            result = result[0]
            # im = result.plot()

            if (keypoints := result.keypoints) is not None:
                for k in keypoints:
                    # conf = k.conf
                    kp = k.data  # x, y, visibility - xy non-normalised
                    kp_dets.append(kp.tolist())
            else:
                continue

            if (boxes := result.boxes) is not None:
                for box in boxes:
                    box_class = int(box.cls[0].item())
                    coords: list[float] = box.xyxy[0].tolist()
                    conf: float = box.conf[0].item()
                    # id: int = box.id
                    dets.append([*coords] + [conf, box_class])
            else:
                continue

            np_dets = np.array(dets)
            np_kp_dets = np.array(kp_dets)

            if len(np_dets) == 0:
                np_dets = np.empty((0, 6))

            tracks: npt.NDArray[np.int_ | np.float_] = tracker.update(np_dets, im)

            if tracks.shape[0] != 0:
                inds = tracks[:, 7].astype("int")  # float64 to int
                kps = np_kp_dets[inds]

                pose_plotter.plot_keypoints(image=im, keypoints=kps)
                _ = tracker.plot_results(im, show_trajectories=True)

            cv2.imshow(f"{tracker.__class__} detection", im)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") or key == ord("q"):
                break

    vid.release()
    cv2.destroyAllWindows()


def main(args: DefaultArguments):
    args.classes = 0

    yolo_model = YOLO("yolov8s-pose.pt")
    tracker = DeepOCSORT(
        model_weights=Path("osnet_x0_25_msmt17.pt"), device="cuda:0", fp16=True
    )

    plot_pose_from_video(args.source, yolo_model, tracker)


if __name__ == "__main__":
    args = get_opts()
    main(args)
