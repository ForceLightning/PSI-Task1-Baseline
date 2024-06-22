from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import json
import os
from typing import Any

import torch
from torch import nn
from typing_extensions import Self


@dataclass
class ResumeTrainingInfo:
    """Contains information required to resume training with methods to initialise and save the
        model's and optimizer's state dicts.

    :param num_epochs: Total number of epochs for training.
    :type num_epochs: int
    :param last_epoch: The current epoch to run.
    :type last_epoch: int
    """

    num_epochs: int
    current_epoch: int

    def dump_resume_info(self, checkpoint_path: str | os.PathLike[Any]):
        """Dumps training resumption info.

        :param checkpoint_path: Path to checkpoint directory.
        :type checkpoint_path: str | os.PathLike
        """
        with open(
            os.path.join(checkpoint_path, "resume.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load_resume_info(
        cls, checkpoint_path: str | os.PathLike[Any]
    ) -> ResumeTrainingInfo | None:
        """Loads training resumption info.

        :param checkpoint_path: Path to checkpoint directory.
        :type checkpoint_path: str | os.PathLike

        :return: Training resumption info if it exists, otherwise None.
        :rtype: ResumeTrainingInfo | None

        :raises: FileNotFoundError: If the `checkpoint_path` is invalid.
        """
        if os.path.exists(resume_path := os.path.join(checkpoint_path, "resume.json")):
            with open(resume_path, "r", encoding="utf-8") as f:
                data: dict[{"num_epochs": int, "current_epoch": int}] = json.load(f)
                assert all(
                    x in data.keys() for x in ["num_epochs", "current_epoch"]
                ), "malformed training resumption data"
                return cls(data["num_epochs"], data["current_epoch"])

        return None

    def load_state_dicts(
        self,
        checkpoint_path: str | os.PathLike[Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Loads state dicts for the model and optimizer.

        :param checkpoint_path: Path to checkpoint directory.
        :type checkpoint_path: str | os.PathLike
        :param model: Model for loading the `state_dict` into, must be the same class as the saved
            checkpoint.
        :type model: nn.Module
        :param optimizer: Optimizer for loading the `state_dict` into, must be the same class as the
            saved checkpoint.
        :type optimizer: torch.optim.Optimizer
        :param scheduler: Scheduler for loading the `state_dict` into, must be the same class as the saved checkpoint.
        :rtype: tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        """
        if os.path.exists(model_ckpt := os.path.join(checkpoint_path, "latest.pth")):
            model.load_state_dict(torch.load(model_ckpt))
        if os.path.exists(
            optim_ckpt := os.path.join(checkpoint_path, "latest_optim.pth")
        ):
            optimizer.load_state_dict(torch.load(optim_ckpt))
        if os.path.exists(
            sched_ckpt := os.path.join(checkpoint_path, "latest_sched.pth")
        ):
            scheduler.load_state_dict(torch.load(sched_ckpt))

        return model, optimizer, scheduler
