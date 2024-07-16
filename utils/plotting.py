import ast
import json
import math
import os
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import torch
from cv2 import typing as cvt
from matplotlib import pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from numpy import typing as npt


class PipelinePlotter:
    joints_dict: dict[
        str,
        dict[
            {
                "keypoints": dict[int, str],
                "skeleton": list[tuple[int, int]],
            }
        ],
    ] = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle",
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                (15, 13),
                (13, 11),
                (16, 14),
                (14, 12),
                (11, 12),
                (5, 11),
                (6, 12),
                (5, 6),
                (5, 7),
                (6, 8),
                (7, 9),
                (8, 10),
                (1, 2),
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),  # (3, 5), (4, 6)
                (0, 5),
                (0, 6),
            ],
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist",
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                (5, 4),
                (4, 3),
                (0, 1),
                (1, 2),
                (3, 2),
                (3, 6),
                (2, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (13, 7),
                (12, 7),
                (13, 14),
                (12, 11),
                (14, 15),
                (11, 10),
            ],
        },
    }

    @classmethod
    def draw_points_and_skeleton(
        cls,
        image: npt.NDArray[np.uint8] | cvt.MatLike,
        points: npt.NDArray[np.float_],
        skeleton: Literal["coco", "mpii"] = "coco",
        points_color_palette: str = "tab20",
        points_palette_samples: int = 16,
        skeleton_color_palette: str = "Set2",
        skeleton_palette_samples: int = 8,
        person_index: int = 0,
        confidence_threshold: float = 0.5,
        flip_xy: bool = False,
    ) -> npt.NDArray[np.uint8] | cvt.MatLike:
        image = cls.draw_skeleton(
            image,
            points,
            cls.joints_dict[skeleton]["skeleton"],
            skeleton_color_palette,
            skeleton_palette_samples,
            person_index,
            confidence_threshold,
            flip_xy,
        )
        image = cls.draw_points(
            image,
            points,
            points_color_palette,
            points_palette_samples,
            confidence_threshold,
            flip_xy,
        )
        return image

    @classmethod
    def draw_skeleton(
        cls,
        image: npt.NDArray[np.uint8] | cvt.MatLike,
        points: npt.NDArray[np.float_],
        skeleton: list[tuple[int, int]],
        color_palette: str = "Set2",
        palette_samples: int = 8,
        person_index: int = 0,
        confidence_threshold: float = 0.5,
        flip_xy: bool = False,
    ) -> npt.NDArray[np.uint8] | cvt.MatLike:
        try:
            colors = (
                np.round(np.array(plt.get_cmap(color_palette).colors) * 255)
                .astype(np.uint8)[:, ::-1]
                .tolist()
            )
        except AttributeError:  # if palette has not pre-defined colors
            colors = (
                np.round(
                    np.array(
                        plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
                    )
                    * 255
                )
                .astype(np.uint8)[:, -2::-1]
                .tolist()
            )

        for i, joint in enumerate(skeleton):
            pt1, pt2 = points[list(joint)]
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                image = cv2.line(
                    image,
                    (
                        (int(pt1[0]), int(pt1[1]))
                        if flip_xy
                        else (int(pt1[1]), int(pt1[0]))
                    ),
                    (
                        (int(pt2[0]), int(pt2[1]))
                        if flip_xy
                        else (int(pt2[1]), int(pt2[0]))
                    ),
                    tuple(colors[person_index % len(colors)]),
                    2,
                )

        return image

    @classmethod
    def draw_points(
        cls,
        image: npt.NDArray[np.uint8] | cvt.MatLike,
        points: npt.NDArray[np.float_],
        color_palette: str = "tab20",
        palette_samples: int = 16,
        confidence_threshold: float = 0.5,
        flip_xy: bool = True,
    ) -> npt.NDArray[np.uint8] | cvt.MatLike:
        try:
            colors = (
                np.round(np.array(plt.get_cmap(color_palette).colors) * 255)
                .astype(np.uint8)[:, ::-1]
                .tolist()
            )
        except AttributeError:  # if palette has not pre-defined colors
            colors = (
                np.round(
                    np.array(
                        plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
                    )
                    * 255
                )
                .astype(np.uint8)[:, -2::-1]
                .tolist()
            )

        circle_size = max(1, min(image.shape[:2]) // 160)

        for i, pt in enumerate(points):
            if pt[2] > confidence_threshold:
                image = cv2.circle(
                    image,
                    (int(pt[0]), int(pt[1])) if flip_xy else (int(pt[1]), int(pt[0])),
                    circle_size,
                    tuple(colors[i % len(colors)]),
                    -1,
                )

        return image


class PosePlotter:
    def __init__(self):
        self.pose_palette: npt.NDArray[np.uint8] = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        self.skeleton: list[tuple[int, int]] = [
            (16, 14),
            (14, 12),
            (17, 15),
            (15, 13),
            (12, 13),
            (6, 12),
            (7, 13),
            (6, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (9, 11),
            (2, 3),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
        ]
        self.limb_color: npt.NDArray[np.uint8] = self.pose_palette[
            [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
        ]
        self.kpt_color: npt.NDArray[np.uint8] = self.pose_palette[
            [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
        ]

    def plot_keypoints(
        self,
        image: cvt.MatLike,
        keypoints: npt.NDArray[np.float_],
        shape: tuple[int, int] = (640, 640),
        radius: int = 5,
    ):
        assert (
            keypoints.ndim == 4
        ), f"Expected 3D array, got {keypoints.ndim}D array of shape {keypoints.shape}"
        for kpt in keypoints:  # tracks
            for point in kpt:  # individual keypoints
                for i, p in enumerate(point):
                    color_k = [int(x) for x in self.kpt_color[i]]
                    x_coord = int(p[0])
                    y_coord = int(p[1])
                    conf = p[2]
                    if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                        if conf < 0.5:
                            continue
                        _ = cv2.circle(
                            image,
                            (x_coord, y_coord),
                            radius=radius,
                            color=color_k,
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )

        for kpt in keypoints:
            for point in kpt:
                ndim = point.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    pos1 = (int(point[(sk[0] - 1), 0]), int(point[(sk[0] - 1), 1]))
                    pos2 = (int(point[(sk[1] - 1), 0]), int(point[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = point[(sk[0] - 1), 2]
                        conf2 = point[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if (
                        pos1[0] % shape[1] == 0
                        or pos1[1] % shape[0] == 0
                        or pos1[0] < 0
                        or pos1[1] < 0
                    ):
                        continue
                    if (
                        pos2[0] % shape[1] == 0
                        or pos2[1] % shape[0] == 0
                        or pos2[0] < 0
                        or pos2[1] < 0
                    ):
                        continue
                    _ = cv2.line(
                        image,
                        pos1,
                        pos2,
                        [int(x) for x in self.limb_color[i]],
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

    def plot_masks(self, image: cvt.MatLike, masks):
        for mask in masks:
            contours = np.array(mask, dtype=np.int32)
            contours = contours.reshape((-1, 1, 2))
            cv2.drawContours(image, [contours], -1, (0, 255, 0), cv2.FILLED)


def draw_landmarks_on_image(rgb_image: cvt.MatLike, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def region_of_interest(
    img: cvt.MatLike, vertices: Sequence[cvt.MatLike]
) -> cvt.MatLike:
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_the_lines(
    img: cvt.MatLike,
    lines: list[list[list[int]]] | npt.NDArray[np.float_] | None,
) -> cvt.MatLike | None:
    if lines is None:
        return

    imge = np.copy(img)
    blank_image = np.zeros((imge.shape[0], imge.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            _ = cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            imge = cv2.addWeighted(imge, 0.8, blank_image, 1, 0.0)
    return imge


def road_lane_detection(orig_image: cvt.MatLike) -> cvt.MatLike | None:
    # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = orig_image
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_coor = [(0, 600), (width / 2, height / 2), (width, 600)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)

    cropped = region_of_interest(
        canny_image, np.array([region_of_interest_coor], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25,
    )
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    print(lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
        if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
            continue
        if slope <= 0:  # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else:  # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5))  # <-- Just below the horizon
    max_y = int(image.shape[0])  # <-- The bottom of the image
    if (
        left_line_x == []
        or left_line_y == []
        or right_line_x == []
        or right_line_y == []
    ):
        return orig_image

    poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    image_with_lines = draw_the_lines(
        image,
        [
            [
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]
        ],
    )

    return image_with_lines


def overlay(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.float_ | np.int_],
    color: tuple[int, int, int],
    alpha: float = 0.5,
    resize: tuple[int, int] | None = None,
) -> cvt.MatLike:
    """Combines image and its segmentation mask into a single image.

    :param npt.NDArray[np.uint8] image: Training image.
    :param npt.NDArray[np.float_ | np.int_] mask: Segmentation mask.
    :param tuple[int, int, int] color: Color for segmentation mask rendering.
    :param float alpha: Segmentation mask transparency, defaults to 0.5.
    :param resize: If provided, both image and its mask are resized before blending them
    together.
    :type resize: tuple[int, int] or None
    :returns: Combined image and masks.
    :rtype: cvt.MatLike
    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def get_video_dimensions(video_path: str) -> tuple[None, None] | tuple[int, int]:
    """
    Get height and width of the video

    :param str video_path: Path to the video file.
    :returns: Tuple of width, height if exists, otherwise None, None.
    :rtype: tuple[None, None] or tuple[int, int]
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return None, None

    # Get the video frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()

    return width, height


def consolidate_yolo_data(
    video_id: int,
) -> tuple[
    dict[str, list[tuple[float, float, float, float]]], dict[str, list[str]], int
]:
    """
    Consolidate the output data from YOLO

    :param int video_id: Video file ID.
    :returns: Tuple of bounding boxes, frames, and video IDs.
    :rtype: tuple[dict[str, list[tuple[float, float, float, float]]], dict[str, list[str]], int]
    """
    bbox_holder: dict[str, list[tuple[float, float, float, float]]] = {}
    frames_holder: dict[str, list[str]] = {}
    yolo_data: str = os.path.join(os.getcwd(), video_id + ".txt")

    with open(yolo_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split(" ")
            track_id = elements[0]
            x1 = float(elements[1])
            y1 = float(elements[2])
            x2 = float(elements[3])
            y2 = float(elements[4])
            frame_id = elements[-1].strip()

            if track_id not in bbox_holder:
                bbox_holder[track_id] = [(x1, y1, x2, y2)]
            else:
                bbox_holder[track_id].append((x1, y1, x2, y2))
            if track_id not in frames_holder:
                frames_holder[track_id] = [frame_id]
            else:
                frames_holder[track_id].append(frame_id)
            # To prevent error when YOLO gives iffy predictions
            # try:
            #     track_id = elements[5].strip()
            #     track_id = "track_" + track_id
            #     if track_id not in bbox_holder:
            #         bbox_holder[track_id] = [[xtl, ytl, xbr, ybr]]
            #     else:
            #         bbox_holder[track_id].append([xtl, ytl, xbr, ybr])
            #     if track_id not in frames_holder:
            #         frames_holder[track_id] = [frame_id]
            #     else:
            #         frames_holder[track_id].append(frame_id)
            # except:
            #     pass

    return bbox_holder, frames_holder, video_id


# Make folder with the data then create dataset from there
def save_data_to_txt(
    bbox_dict: dict[str, list[tuple[float, float, float, float]]],
    frames_dict: dict[str, list[str]],
    video_id: int,
) -> None:
    """
    Make folder with the data then create dataset from there

    :param bbox_dict: Dictionary containing the pedestrian ID and the bounding boxes.
    :type bbox_dict: dict[str, list[tuple[float, float, float, float]]]
    :param dict[str, list[str]] frames_dict: Dictionary containing the pedestrian ID
    and the frames.
    :param int video_id: The video ID.
    :returns: None
    """
    data_folder = os.path.join(os.getcwd(), "yolo_results_data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    file_count = 0
    for k, v in bbox_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 15 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "w") as f:
                    _ = f.write(f"{video_id}\t{k}\t{v[i : i + 15]}")
                file_count += 1
    file_count = 0
    for k, v in frames_dict.items():
        if len(v) >= 15:
            for i in range(len(v) - 15 + 1):
                file_name = os.path.join(data_folder, str(file_count + 1) + ".txt")
                with open(file_name, "a") as f:
                    _ = f.write(f"\t{v[i : i + 15]}")
                file_count += 1


def visualise_annotations(annotation_path: str, selected_frame: int) -> None:
    """
    Visualise the bounding boxes that are fed into TCN model for sanity check

    :param str annotation_path: Path to text with labels for Pedestrian Intent
    Prediction.
    :param int selected_frame: The frame to visualise (0..=14)
    :returns: None
    """
    # Read annotations
    with open(annotation_path, "r") as f:
        annotations = f.readlines()

    for ann in annotations:
        elements = ann.split("\t")
        vid_id = elements[0]
        ped_id = elements[1]
        bboxes: list[tuple[float, float, float, float]] = ast.literal_eval(elements[2])
        bbox = bboxes[selected_frame]
        xtl = int(bbox[0])
        ytl = int(bbox[1])
        xbr = int(bbox[2])
        ybr = int(bbox[3])
        frames = ast.literal_eval(elements[3])
        frame = str(frames[selected_frame]).zfill(3)

        # Load the image
        image_path = os.path.join(os.getcwd(), "frames", vid_id, frame + ".jpg")
        image = cv2.imread(image_path)

        # Draw bounding box
        _ = cv2.rectangle(image, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)
        _ = cv2.putText(
            img=image,
            text=ped_id,
            org=(xbr, ybr),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
        )

    # Display the image
    cv2.imshow("Annotated Image", image)
    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualise_intent(annotation_path: str, intent_path: str) -> None:
    """
    Visualise the pedestrian intent (Bbox is 1 frame behind)

    :param str annotation_path: Path to text with YOLO annotations.
    :param str intent_path: Path to text with labels for Pedestrian Intent Prediction.
    :returns: None
    """

    with open(intent_path, "r") as f:
        intent_results = f.read()

    # Parse JSON data
    parsed_data: dict[
        str,
        dict[
            str,
            dict[
                str, dict[{"intent": int, "intent_prob": float, "disagreement": float}]
            ],
        ],
    ] = json.loads(intent_results)

    for vid_id, tracks in parsed_data.items():
        print("Video:", vid_id)
        for track, frames in tracks.items():
            print("Track:", track)
            for frame, details in frames.items():
                yolo_frame = int(frame)
                with open(annotation_path, "r") as f:
                    annotations = f.readlines()
                for ann in annotations:
                    element = ann.split(" ")

                    if ("track_" + element[0]) == track and element[
                        -1
                    ].strip() == frame:
                        xtl = float(element[1])
                        ytl = float(element[2])
                        xbr = float(element[3])
                        ybr = float(element[4])

                        frame = frame.zfill(3)
                        intent = details["intent"]
                        image_path = os.path.join(
                            os.getcwd(), "frames", vid_id, frame + ".jpg"
                        )
                        image = cv2.imread(image_path)
                        print(
                            "GT Frame:",
                            frame,
                            "| Intent:",
                            intent,
                            "| Bbox Frame:",
                            yolo_frame,
                        )
                        _ = cv2.rectangle(
                            image,
                            (int(xtl), int(ytl)),
                            (int(xbr), int(ybr)),
                            (255, 0, 0),
                            2,
                        )
                        _ = cv2.putText(
                            img=image,
                            text=str(intent),
                            org=(int(xbr), int(ybr)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                        )
                        _ = cv2.putText(
                            img=image,
                            text=track,
                            org=(int(xbr), int(ytl)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                        )
                        cv2.imshow("Annotated Image", image)
                        _ = cv2.waitKey(0)
                        cv2.destroyAllWindows()
