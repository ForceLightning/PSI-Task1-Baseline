from __future__ import annotations

import os
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from cv2 import typing as cvt
from numpy import typing as npt
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import default_collate
from torchvision.transforms import v2
from tqdm.auto import tqdm

from data.custom_dataset import T_drivingBatch, T_drivingSample
from eval import load_args, load_model_state_dict
from models.build_model import build_model
from opts import init_args
from utils.args import DefaultArguments
from utils.cuda import *
from utils.track2_inference_pipeline import DirtyGotoException, id_to_color


def plot_decision_on_img(
    im: npt.NDArray[np.uint8] | cvt.MatLike,
    speed_logits: torch.Tensor,
    direction_logits: torch.Tensor,
) -> npt.NDArray[np.uint8] | cvt.MatLike:
    h, w, _ = im.shape

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(im)

    font = ImageFont.truetype("seguisym.ttf", size=60)
    draw = ImageDraw.Draw(im_pil, "RGB")

    default_speed = ["⇩", "⚪", "⇧"]
    speed_options = ["⚫", "⬇", "⬆"]

    default_direction = ["⇦", "⇧", "⇨"]
    direction_options = ["⬆", "⬅", "➡"]

    pred_to_fmt_index = [1, 0, 2]

    speed_pred: int = speed_logits.argmax().item()
    speed_proba: float = speed_logits.softmax(dim=-1).max().item()
    default_speed[pred_to_fmt_index[speed_pred]] = speed_options[speed_pred]

    direction_pred: int = direction_logits.argmax().item()
    direction_proba: float = direction_logits.softmax(dim=-1).max().item()
    default_direction[pred_to_fmt_index[direction_pred]] = direction_options[
        direction_pred
    ]

    speed_str = "\n".join(default_speed[::-1])
    direction_str = " ".join(default_direction)

    speed_color = id_to_color(1)
    dir_color = id_to_color(0)

    draw.text(
        (0, h - 240),
        speed_str,
        font=font,
        fill=speed_color,
    )

    draw.text(
        (w - 180, h - 120),
        direction_str,
        font=font,
        fill=dir_color,
    )

    im = np.asarray(im_pil)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    im = cv2.putText(
        im,
        f"speed_conf: {speed_proba:.2f}",
        (60, h - 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        speed_color,
        2,
    )

    im = cv2.putText(
        im,
        f"direction_conf: {direction_proba:.2f}",
        (w - 180, h - 130),  # top left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        dir_color,
        2,
    )

    return im


def infer(
    args: DefaultArguments,
    model: nn.Module | nn.DataParallel,
    source: str | Path,
    transform: v2.Compose | None = None,
    save_video: bool = False,
    save_path: str | Path = "track3_inference.py",
):
    _ = model.eval()

    source = source if isinstance(source, str) else str(source)
    source = os.path.normpath(source)

    vid = cv2.VideoCapture(source)
    frame_number = 0

    buffer: deque[cvt.MatLike | torch.Tensor] = deque(maxlen=args.observe_length)
    frame_list: deque[int] = deque(maxlen=args.observe_length)

    video_writer: cv2.VideoWriter | None = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video_id = os.path.normpath(source).split(os.sep)[-1]
    video_id = os.path.splitext(video_id)[0]

    with tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), leave=True) as pbar:
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
                frame_list.append(frame_number)
                buffer.append(im if transform is None else transform(im))

                if len(buffer) < args.observe_length:
                    raise DirtyGotoException("Buffer is not full yet")

                global_featmaps: list[torch.Tensor] | torch.Tensor = []

                if args.freeze_backbone:
                    for frame_id in frame_list:
                        global_path = os.path.join(
                            args.dataset_root_path,
                            "features",
                            args.backbone,
                            "global_feats",
                            video_id,
                        )
                        global_featmap: npt.NDArray[np.float_] = np.load(
                            f"{global_path}/{frame_id:03d}.npy"
                        )
                        global_featmaps.append(torch.from_numpy(global_featmap))
                    global_featmaps = torch.stack(global_featmaps).to(DEVICE)

                sample: T_drivingSample = {
                    "video_id": video_id,
                    "frames": torch.tensor(frame_list, dtype=torch.long).unsqueeze(0),
                    "image": torch.stack(tuple(buffer)).to(DEVICE),
                    "global_featmaps": global_featmaps,
                    "local_featmaps": [],
                    "label_speed": 0,
                    "label_direction": 0,
                    "label_speed_prob": 0.0,
                    "label_direction_prob": 0.0,
                }

                batch_tensor: T_drivingBatch = default_collate([sample])

                pred_speed_logit, pred_dir_logit = model(batch_tensor)

                im = plot_decision_on_img(im, pred_speed_logit, pred_dir_logit)

            except DirtyGotoException:
                continue
            except Exception as e:
                pbar.write(
                    f"Error {e} at frame {frame_number:04d}, {traceback.format_exc()}"
                )
                break
            finally:
                cv2.imshow("Pedestrian Intent Prediction Pipeline", im)

                if save_video and video_writer:
                    video_writer.write(im)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") or key == ord("q"):
                    break

            frame_number += 1

    if save_video and video_writer:
        video_writer.release()

    vid.release()
    cv2.destroyAllWindows()


def main(args: DefaultArguments):
    freeze_backbone = args.freeze_backbone
    source = args.source
    save_output: bool = args.save_output
    output: str = args.output

    args.batch_size = 1
    args.driving_loss = ["cross_entropy"]
    args.model_name = "restcn_driving_global"
    args.loss_weights = {
        "loss_intent": 0.0,
        "loss_traj": 0.0,
        "loss_driving": 0.0,
    }
    args.load_image = True
    args.backbone = "resnet50"
    args.observe_length = 15
    args.predict_length = 1
    args.max_track_size = args.observe_length + args.max_track_size

    args.n_layers = 4
    args.kernel_size = 2

    model, _, _ = build_model(args)
    if args.compile_model:
        model = torch.compile(
            model, options={"triton.cudagraphs": True}, fullgraph=True
        )
    model = nn.DataParallel(model)

    if args.checkpoint_path != "ckpts":
        args = load_args(args.dataset_root_path, args.checkpoint_path)
        args.freeze_backbone = freeze_backbone
        model = load_model_state_dict(args.checkpoint_path, model)

    transform = v2.Compose(
        [
            v2.ToPILImage(),
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    infer(args, model, source, transform, save_output, output)


if __name__ == "__main__":
    # args = get_opts()
    parser = init_args()
    _ = parser.add_argument("--save_output", "-so", action="store_true")
    _ = parser.add_argument("--output", "-o", type=str, default="pipeline.avi")
    args = parser.parse_args()

    main(args)
