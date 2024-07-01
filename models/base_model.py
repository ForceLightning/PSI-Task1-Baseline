import abc
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

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
