import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from fastprogress.fastprogress import master_bar, progress_bar
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.transforms import v2

from data.prepare_data import get_dataloader
from database.create_database import create_database
from opts import get_opts
from utils.args import DefaultArguments

CUDA: bool = True if torch.cuda.is_available() else False
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor


class ResNet50_EmbeddingsOnly(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        y = self.resnet(x)
        z = y.view(y.size(0), -1)  # flatten output of resnet
        return z


class VideoDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.X = []
        self.transforms = transforms
        self._set_transforms()
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                self.X.append(
                    {
                        "video_id": root.split(os.sep)[-1],
                        "frame": os.path.splitext(file)[0],
                        "path": os.path.join(root, file),
                    }
                )

    def __getitem__(self, index):
        img_path = self.X[index]["path"]
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return {
            "image": img,
            "video_id": self.X[index]["video_id"],
            "frames": int(self.X[index]["frame"]),
        }

    def _set_transforms(self):
        if self.transforms is None:
            resize_size = 256
            self.transforms = v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(resize_size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.X)


def infer(
    model: nn.Module,
    backbone_name: str,
    dl: DataLoader,
    args: DefaultArguments,
) -> None:
    model.eval()
    mb = master_bar(dl)
    for itern, data in enumerate(mb):
        embeddings = model(torch.squeeze(data["image"]).to(DEVICE))
        embeddings = embeddings.detach().cpu().numpy()
        for i, fid in enumerate(data["frames"]):
            embedding = embeddings[i]
            video_id = data["video_id"][i]
            global_path = os.path.join(
                args.dataset_root_path,
                "features",
                backbone_name,
                "global_feats",
                video_id,
            )
            mb.main_bar.comment = f"{itern}/{len(dl)}, {global_path}/{fid:03d}.npy"
            if not os.path.exists(global_path):
                os.makedirs(global_path)
            with open(f"{global_path}/{fid:03d}.npy", "wb") as f:
                np.save(f, embedding)


def main(backbone: str, args: DefaultArguments):
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")
    # train_loader, val_loader, test_loader = get_dataloader(args)
    ds = VideoDataset("./frames")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=8,
    )

    # Load model
    model = None
    match backbone:
        case "resnet50":
            # resnet= resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
            # del model._modules["fc"]
            model = ResNet50_EmbeddingsOnly().to(DEVICE)
        # TODO: Implement VGG16 and FasterRCNN.
        # case "vgg16":
        # model = vgg16(weights=VGG16_Weights.DEFAULT).to(DEVICE)
        # del model._modules["classifier"]
        # case "faster_rcnn":
        # model = fasterrcnn_resnet50_fpn_v2(
        # weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        # ).to(DEVICE)
        # del model._modules["roi_heads"]._modules["box_predictor"]
        case str(x):
            raise NotImplementedError(f"{x} not implemented as a backbone yet")

    infer(model, backbone, dl, args)


if __name__ == "__main__":
    args = get_opts()
    args.task_name = args.task_name if args.task_name else "ped_traj"
    args.persist_dataloader = False

    match args.task_name:
        case "ped_intent":
            args.database_file = "intent_database_train.pkl"
            args.intent_model = True
            args.batch_size = 256
            # intent prediction
            args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
            args.intent_type = "mean"  # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
            args.intent_loss = ["bce"]
            args.intent_disagreement = (
                1  # -1: not use disagreement 1: use disagreement to reweigh samples
            )
            args.intent_positive_weight = (
                0.5  # Reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)
            )
            args.predict_length = 0  # only make one intent prediction
            args.model_name = "tcn_int_bbox"
            args.loss_weights = {
                "loss_intent": 1.0,
                "loss_traj": 0.0,
                "loss_driving": 0.0,
            }
            args.load_image = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
        case "ped_traj":
            args.database_file = "traj_database_train.pkl"
            args.intent_model = False  # if (or not) use intent prediction module to support trajectory prediction
            args.traj_model = True
            args.traj_loss = ["bbox_l1"]
            # args.batch_size = [256]
            args.batch_size = 32
            args.predict_length = 0
            # args.model_name = "tcn_traj_bbox"
            # args.model_name = "tcn_traj_bbox_int"
            args.model_name = "tcn_traj_global"
            args.loss_weights = {
                "loss_intent": 0.0,
                "loss_traj": 1.0,
                "loss_driving": 0.0,
            }
            args.load_image = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
        case "driving_decision":
            args.database_file = "driving_database_train.pkl"
            args.driving_loss = ["cross_entropy"]
            args.batch_size = 64
            args.model_name = "reslstm_driving_global"
            args.loss_weights = {
                "loss_intent": 0.0,
                "loss_traj": 0.0,
                "loss_driving": 1.0,
            }
            args.load_image = True
            args.seq_overlap_rate = 1  # overlap rate for train/val set
            args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE

    args.observe_length = 1
    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = "enlarge"
    args.normalize_bbox = None

    main(args.backbone, args)
