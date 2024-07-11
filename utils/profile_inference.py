import os
from random import randint
from typing import Literal

import torch
import torch._dynamo
from torch import nn
from tqdm.auto import tqdm

from data.custom_dataset import T_drivingBatch, T_intentBatch
from models.build_model import build_model
from opts import get_opts
from utils.args import DefaultArguments
from utils.cuda import *

torch._dynamo.config.suppress_errors = True

TASK_MODELS: dict[Literal["ped_intent", "ped_traj", "driving_decision"], list[str]] = {
    "ped_intent": ["lstm_int_bbox", "tcn_int_bbox"],
    "ped_traj": [
        "lstm_traj_bbox",
        "tcn_traj_bbox",
        "tcn_traj_bbox_int",
        "tcn_traj_global",
        "tcan_traj_bbox",
        "tcan_traj_global",
        "transformer_traj_bbox",
        "transformer_traj_bbox_pose",
        "transformer_traj_intent_bbox_pose",
    ],
    "driving_decision": ["reslstm_driving_global", "restcn_driving_global"],
}


def get_batchlike_data(
    task: Literal["ped_intent", "ped_traj", "driving_decision"],
    batch_size: int,
    observe_length: int,
    predict_length: int,
) -> T_intentBatch | T_drivingBatch:
    seq_length = observe_length + predict_length
    data: T_intentBatch | T_drivingBatch
    if task in ["ped_intent", "ped_traj"]:
        intent_batch: T_intentBatch = {
            "global_featmaps": torch.randn(batch_size, seq_length, 2048).to(DEVICE),
            "local_featmaps": [],
            "image": torch.randn(batch_size, seq_length, 3, 224, 224).to(DEVICE),
            "bboxes": torch.randn(batch_size, seq_length, 4).to(DEVICE),
            "original_bboxes": torch.randn(batch_size, seq_length, 4).to(DEVICE),
            "intention_binary": torch.randint(0, 1, (batch_size, seq_length)).to(
                DEVICE
            ),
            "intention_prob": torch.randn(batch_size, seq_length).to(DEVICE),
            "frames": torch.randint(0, 1, (batch_size, 1))
            .expand((batch_size, observe_length))
            .to(DEVICE),
            "total_frames": torch.randint(0, 1, (batch_size, 1))
            .expand((batch_size, seq_length))
            .to(DEVICE),
            "video_id": [[f"video_{randint(0, 204):03d}"] * seq_length] * batch_size,
            "ped_id": [[f"track_{randint(0, 204):03d}"] * seq_length] * batch_size,
            "disagree_score": torch.randn(batch_size, seq_length).to(DEVICE),
            "pose": torch.randn(batch_size, seq_length, 17, 2).to(DEVICE),
            "pose_masks": torch.randint(0, 1, (batch_size, seq_length, 17)).to(DEVICE),
        }
        data = intent_batch
    else:
        driving_batch: T_drivingBatch = {
            "video_id": [[f"video_{randint(0, 204):03d}"] * seq_length] * batch_size,
            "frames": torch.randint(0, 1, (batch_size, 1))
            .expand((batch_size, observe_length))
            .to(DEVICE),
            "image": torch.randn(batch_size, seq_length, 3, 224, 224).to(DEVICE),
            "global_featmaps": torch.randn(batch_size, seq_length, 2048).to(DEVICE),
            "local_featmaps": [],
            "label_speed": torch.randint(0, 1, (batch_size, seq_length)).to(DEVICE),
            "label_direction": torch.randint(0, 1, (batch_size, seq_length)).to(DEVICE),
            "label_speed_prob": torch.randn(batch_size, seq_length).to(DEVICE),
            "label_direction_prob": torch.randn(batch_size, seq_length).to(DEVICE),
        }
        data = driving_batch

    return data


@torch.no_grad()
def infer(
    args: DefaultArguments,
    model: nn.Module,
    data: T_intentBatch | T_drivingBatch,
    p: torch.profiler.profile,
) -> None:
    _ = model.eval()
    for i in tqdm(range(10), desc="Inference batches", leave=False):
        if "transformer" in args.model_name:
            _ = model.generate(data)  # type: ignore[reportAny]
        else:
            _ = model(data)
        p.step()


def main(args: DefaultArguments):
    args.checkpoint_path = os.path.join(args.dataset_root_path, "ckpts")

    args.observe_length = 15
    args.n_layers = 4
    args.steps_per_epoch = 10

    if "transformer" in args.model_name:
        args.observe_length += 1

    data = get_batchlike_data(
        task=args.task_name,
        batch_size=args.batch_size,
        observe_length=args.observe_length,
        predict_length=args.predict_length,
    )

    mbar = tqdm(total=sum([len(model_list) for model_list in TASK_MODELS.values()]))
    for task, model_list in TASK_MODELS.items():
        args.task_name = task
        if args.task_name in ["ped_intent", "driving_decision"]:
            args.predict_length = 1
        else:
            args.predict_length = 45

        model_list_iter = tqdm(model_list, desc="Models", leave=False)
        for model_name in model_list_iter:
            args.model_name = model_name
            model_list_iter.set_postfix({"Model": model_name})
            model, _, _ = build_model(args)
            # NOTE: Broken on Windows
            # if not any(
            #     unsupported_model in model_name.lower()
            #     for unsupported_model in ["rnn", "gru", "lstm"]
            # ):  # torch.compile does not work with RNNs, GRUs, and LSTMs.
            #     model: nn.Module = torch.compile(
            #         model, options={"triton.cudagraphs": True}, fullgraph=True
            #     )
            model = model.to(DEVICE)

            export_path = os.path.join(
                args.checkpoint_path,
                args.task_name,
                args.dataset,
                args.model_name,
                "profiling_runs",
            )

            if not os.path.exists(export_path):
                os.makedirs(export_path)

            def trace_handler(prof: torch.profiler.profile):
                prof.export_chrome_trace(
                    os.path.join(export_path, f"profiling_results.pt.trace.json")
                )
                mbar.write(f"Model: {model_name}")
                mbar.write(
                    str(prof.key_averages().table(sort_by="self_cpu_time_total"))
                )
                return torch.profiler.tensorboard_trace_handler(export_path)

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=2),
                on_trace_ready=trace_handler,
            ) as p:
                infer(args, model, data, p)

        _ = mbar.update(1)


if __name__ == "__main__":
    args = get_opts()
    args.dataset_root_path = "../"
    main(args)
