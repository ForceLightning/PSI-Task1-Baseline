import numpy as np

from utils.args import DefaultArguments


def save_args(args: DefaultArguments):
    # 3. args
    with open(args.checkpoint_path + "/args.txt", "w", encoding="utf-8") as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):
                val = f"'{val}'"
            f.write("{}: {}\n".format(arg, val))
    np.save(args.checkpoint_path + "/args.npy", args)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val: int | float
        self.avg: int | float
        self.sum: int | float
        self.count: int
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: int | float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
