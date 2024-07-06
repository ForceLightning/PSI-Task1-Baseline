import abc
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    TimeSeriesTransformerModel,
)

from utils.args import DefaultArguments


class IConstructOptimizer(metaclass=abc.ABCMeta):
    """An "interface" that expects concrete classes to implement the `build_optimizer`
    method.
    """

    @abc.abstractmethod
    def build_optimizer(self, args: DefaultArguments) -> tuple[Optimizer, LRScheduler]:
        """
        Build the optimizer and learning rate scheduler for the model.

        Based on the value of arguments passed in during both model initialisation and
        of this method, return instances of the optimizer and lr scheduler to be used
        during training.

        :param DefaultArguments args: Arguments passed during the training process.

        :returns: Optimizer and LR scheduler.
        :rtype: tuple[Optimizer, LRScheduler]

        .. rubric:: Example

        >>> # initialise the model
        >>> model = ModelClass()
        >>> opt, sched = model.build_optimizer(args)
        """
        pass


class ITSTransformerWrapper(metaclass=abc.ABCMeta):
    """An "interface" that expects concrete classes to implement the `generate` method."""

    transformer: TimeSeriesTransformerForPrediction | TimeSeriesTransformerModel
    config: TimeSeriesTransformerConfig

    @abc.abstractmethod
    def generate(self, x: Any) -> torch.Tensor:
        """Generates samples from the probabilistic time series model without the
        :py:meth:`forward` hooks.

        :param Any x: Input data.
        :return: Sample predictions from the model.
        :rtype: torch.Tensor
        """
        pass
