import os
import abc
from copy import deepcopy
import ast

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torchvision.transforms import v2


class VideoDataset(Dataset):
    def __init__(self, data: dict, args: dict, stage="train") -> None:
        super().__init__()
        self.data = data
        self.args = args
        self.stage = stage
        self.set_transform()
        self.images_path = os.path.join(args.dataset_root_path, "frames")

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError("Method not implemented")

    def load_images(
        self, video_ids: list, frame_list: list, bboxes: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads the images from the local storage

        Args:
            video_ids (list): List of video ids
            frame_list (list): List of frame ids
            bboxes (list): List of frames of bounding boxes

        Raises:
            ValueError: Cropped image shape is empty

        Returns:
            tuple[Tensor | list]: Tuple of images and cropped images
        """
        images = []
        cropped_images = []
        video_name = video_ids[0]

        for i, frame_id in enumerate(frame_list):
            bbox = bboxes[i]
            img_path = os.path.join(
                self.images_path, video_name, str(frame_id).zfill(5) + ".png"
            )
            img = self.rgb_loader(img_path)
            original_bbox = deepcopy(bbox)
            bbox = self.jitter_bbox(img, [bbox], self.args.crop_mode, 2.0)[0]
            bbox = self.squarify(bbox, 1, img.shape[1])
            bbox = list(map(int, bbox[0:4]))
            cropped_img = Image.fromarray(img).crop(bbox)
            cropped_img = np.array(cropped_img)
            if not cropped_img.shape:
                raise ValueError("Cropped image shape is empty")
            cropped_img = self.img_pad(cropped_img, mode="pad_resize", size=224)
            cropped_img = np.array(cropped_img)

            if self.transform:
                img, original_bbox = self.transform(img, original_bbox)
                cropped_img, bbox = self.transform(cropped_img, bbox)

            images.append(img)
            cropped_images.append(cropped_img)

        return torch.stack(images), torch.stack(cropped_images)

    def rgb_loader(self, img_path: os.PathLike | str) -> cv2.typing.MatLike:
        """Loads the image in RGB format using OpenCV

        Args:
            img_path (os.PathLike | str): Path to the image

        Returns:
            cv2.typing.MatLike: Image in RGB format
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_reason_features(self, video_ids, ped_ids, frame_list):
        """Loads the reason features from the local storage

        Args:
            video_ids (list): List of video ids
            ped_ids (list): List of pedestrian ids
            frame_list (list): List of frame ids

        Returns:
            Tensor | list: List of reason features
        """
        feature_list = []
        video_name = video_ids[0]
        if "rsn" in self.args.model_name:
            for i, fid in enumerate(frame_list):
                pid = ped_ids[i]
                local_path = os.path.join(
                    self.args.dataset_root_path,
                    "features/bert_description",
                    video_name,
                    pid,
                )
                feature_file = np.load(f"{local_path}/{fid:03d}.npy")
                feature_list.append(torch.tensor(feature_file))

        feature_list = [] if len(feature_list) < 1 else torch.stack(feature_list)
        return feature_list

    def load_features(self, video_ids, ped_ids, frame_list):
        """Loads the features from the local storage

        Args:
            video_ids (list): List of video ids
            ped_ids (list): List of pedestrian ids
            frame_list (list): List of frame ids

        Returns:
            Tuple[Tensor | list]: Tuple of global and local features
        """
        global_featmaps = []
        local_featmaps = []
        video_name = video_ids[0]
        if "global" in self.args.model_name:
            for i, fid in enumerate(frame_list):
                pid = ped_ids[i]

                global_path = os.path.join(
                    self.args.dataset_root_path,
                    "features",
                    self.args.backbone,
                    "global_feats",
                    video_name,
                )
                global_featmap = np.load(f"{global_path}/{fid:03d}.npy")
                global_featmaps.append(torch.tensor(global_featmap))
        elif "ctxt" in self.args.model_name:
            for i, fid in enumerate(frame_list):
                pid = ped_ids[i]
                local_path = os.path.join(
                    self.args.dataset_root_path,
                    "features",
                    self.args.backbone,
                    "context_feats",
                    video_name,
                    pid,
                )
                local_featmap = np.load(f"{local_path}/{fid:03d}.npy")
                local_featmaps.append(torch.tensor(local_featmap))

        global_featmaps = (
            [] if len(global_featmaps) < 1 else torch.stack(global_featmaps)
        )
        local_featmaps = [] if len(local_featmaps) < 1 else torch.stack(local_featmaps)

        return global_featmaps, local_featmaps

    def set_transform(self) -> None:
        """Sets the transformation for the images based on the stage of the dataset"""
        match (self.stage):
            case "train":
                resize_size = 256
                crop_size = 224
                self.transform = v2.Compose(
                    [
                        v2.ToPILImage(),
                        v2.RandomResizedCrop((resize_size, resize_size)),
                        v2.RandomHorizontalFlip(),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            case _:
                resize_size = 256
                crop_size = 224
                self.transform = v2.Compose(
                    [
                        v2.ToPILImage(),
                        v2.Resize((resize_size, resize_size)),
                        v2.CenterCrop(crop_size),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def squarify(self, bbox: list, squarify_ratio: list, img_width: int) -> list:
        """Squarifies the bounding box based on the given ratio (1 for square, rectangle otherwise)

        Args:
            bbox (list): Bounding box coordinates
            squarify_ratio (float): Ratio for squarifying the bounding box
            img_width (int): Width of the image

        Returns:
            list: Squarified bounding box coordinates
        """
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * squarify_ratio - width
        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2
        # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    def jitter_bbox(
        self,
        img: cv2.typing.MatLike | np.ndarray | PIL.Image.Image,
        bbox: list,
        mode: str,
        ratio: float,
    ) -> list:
        """This method jitters the position or dimensions of the bounding box.

        Args:
            img (cv2.typing.MatLike | np.ndarray | PIL.Image.Image): Image to be cropped and/or
            padded
            bbox (list): Bounding box dimensions for cropping
            mode (str): The type of padding or resizing:
                'same' returns the bounding box unchanged
                'enlarge' increases the size of bounding box based on the given ratio
                'random_enlarge' increases the size of bounding box by randomly sampling a
                value in [0,ratio)
                'move' moves the center of the bounding box in each direction based on the given ratio
            ratio (float): The ratio of change relative to the size of the bounding box
                (for modes 'enlarge' and 'random_enlarge')

        Returns:
            list: Jittered bounding box coordinates
        """
        assert mode in [
            "same",
            "enlarge",
            "move",
            "random_enlarge",
            "random_move",
        ], f"mode {mode} is invalid."

        if mode == "same":
            return bbox

        if mode in ["random_enlarge", "enlarge"]:
            jitter_ratio = abs(ratio)
        else:
            jitter_ratio = ratio

        if mode == "random_enlarge":
            jitter_ratio = np.random.random_sample() * jitter_ratio
        elif mode == "random_move":
            # for ratio between (-jitter_ratio, jitter_ratio)
            # for sampling the formula is [a,b), b > a,
            # random_sample * (b-a) + a
            jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

        jit_boxes = []
        for b in bbox:
            bbox_width = b[2] - b[0]
            bbox_height = b[3] - b[1]

            width_change = bbox_width * jitter_ratio
            height_change = bbox_height * jitter_ratio

            if width_change < height_change:
                height_change = width_change
            else:
                width_change = height_change

            if mode in ["enlarge", "random_enlarge"]:
                b[0] = b[0] - width_change // 2
                b[1] = b[1] - height_change // 2
            else:
                b[0] = b[0] + width_change // 2
                b[1] = b[1] + height_change // 2

            b[2] = b[2] + width_change // 2
            b[3] = b[3] + height_change // 2

            # Checks to make sure the bbox is not exiting the image boundaries
            b = self.bbox_sanity_check(img, b)
            jit_boxes.append(b)
        # elif crop_opts['mode'] == 'border_only':
        return jit_boxes

    def bbox_sanity_check(
        self, img: cv2.typing.MatLike | np.ndarray | PIL.Image.Image, bbox: list
    ) -> list:
        """This is to confirm that the bounding boxes are within image boundaries.
        Otherwise, modifications are applied.

        Args:
            img (cv2.typing.MatLike | np.ndarray | PIL.Image.Image): Image to be cropped and/or
            padded
            bbox (list): Bounding box dimensions for cropping

        Returns:
            list: Modified bounding box coordinates
        """

        img_heigth, img_width, _ = img.shape
        if bbox[0] < 0:
            bbox[0] = 0.0
        if bbox[1] < 0:
            bbox[1] = 0.0
        if bbox[2] >= img_width:
            bbox[2] = img_width - 1
        if bbox[3] >= img_heigth:
            bbox[3] = img_heigth - 1
        return bbox

    def img_pad(
        self,
        img: cv2.typing.MatLike | np.ndarray | PIL.Image.Image,
        mode: str = "warp",
        size: int = 224,
    ) -> cv2.typing.MatLike | np.ndarray | PIL.Image.Image:
        """Pads a given image, crops and/or pads a image given the boundries of the box needed.

        Args:
            img (cv2.typing.MatLike | np.ndarray | PIL.Image.Image): Image to be cropped and/or
            padded
            mode (str, optional): The type of padding or resizing. Defaults to "warp".
                - 'warp': crops the bounding box and resize to the output size.
                - 'same': only crops the image.
                - 'pad_same': maintains the original size of the cropped box and pads with zeros.
                - 'pad_resize': crops the image and resize the cropped box in a way that the longer
                edge is equal to the desired output size in that direction while maintaining the
                aspect ratio. The rest of the image is padded with zeros.
                - 'pad_fit': maintains the original size of the cropped box unless the image is
                bigger than the size in which case it scales the image down, and then pads it.
            size (int, optional): The desired size of output. Defaults to 224.

        Returns:
            cv2.typing.MatLike | np.ndarray | PIL.Image.Image: Padded image
        """
        assert mode in [
            "same",
            "warp",
            "pad_same",
            "pad_resize",
            "pad_fit",
        ], f"Pad mode {mode} is invalid"
        image = img.copy()
        if mode == "warp":
            warped_image = image.resize((size, size), Image.NEAREST)
            return warped_image
        if mode == "same":
            return image
        if mode in ["pad_same", "pad_resize", "pad_fit"]:
            # size is in (width, height)
            img_size = (image.shape[0], image.shape[1])
            ratio = float(size) / max(img_size)
            if mode == "pad_resize" or (
                mode == "pad_fit" and (img_size[0] > size or img_size[1] > size)
            ):
                img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))
                try:
                    image = Image.fromarray(image)
                    image = image.resize(img_size, Image.NEAREST)
                except Exception as e:
                    print("Error from np-array to Image: ", image.shape)
                    print(e)

            padded_image = PIL.Image.new("RGB", (size, size))
            padded_image.paste(
                image, ((size - img_size[0]) // 2, (size - img_size[1]) // 2)
            )
            return padded_image


class PedestrianIntentDataset(VideoDataset):
    def __getitem__(self, index) -> dict:
        """Gets the item with the given index

        Args:
            index (int): Index of the item

        Raises:
            NotImplementedError: L2 normalization not implemented yet

        Returns:
            dict: Data at the given index
        """
        video_ids = self.data["video_id"][index]
        ped_ids = self.data["ped_id"][index]
        assert all(video_ids[0] == vid for vid in video_ids)
        assert all(ped_ids[0] == pid for pid in ped_ids)
        frame_list = self.data["frame"][index][: self.args.observe_length]
        bboxes = self.data["bbox"][index]
        intention_binary = self.data["intention_binary"][index]
        intention_prob = self.data["intention_prob"][index]

        disagree_score = self.data["disagree_score"][index]

        assert len(bboxes) == self.args.max_track_size
        assert len(frame_list) == self.args.observe_length

        global_featmaps, local_featmaps = self.load_features(
            video_ids, ped_ids, frame_list
        )
        reason_features = self.load_reason_features(video_ids, ped_ids, frame_list)

        for frame in range(len(frame_list)):
            bbox = bboxes[frame]
            xtl, ytl, xrb, yrb = bbox  # xtl: x top left, xrb: x right bottom

            if self.args.task_name == "ped_intent" or self.args.task_name == "ped_traj":
                bboxes[frame] = [xtl, ytl, xrb, yrb]

        match (self.args.normalize_bbox):
            case "L2":
                raise NotImplementedError("L2 normalization not implemented yet")
            case "subtract_first_frame":
                bboxes = bboxes - bboxes[:1, :]
            case _:
                pass

        data = {
            "global_featmaps": global_featmaps,
            "local_featmaps": local_featmaps,
            "reason_features": reason_features,
            "bboxes": bboxes,
            "intention_binary": intention_binary,
            "intention_prob": intention_prob,
            "frames": np.array([int(frame) for frame in frame_list]),
            "video_id": video_ids[0],
            "ped_id": ped_ids[0],
            "disagree_score": disagree_score,
        }

        return data

    def __len__(self):
        return len(self.data["frame"])


class DrivingDecisionDataset(VideoDataset):
    def __getitem__(self, index: int):
        """Gets the item with the given index

        Args:
            index (int): Index of the item
        """
        video_ids = self.data["video_id"][index]
        frame_list = self.data["frame"][index][
            : self.args.observe_length
        ]  # return first 15 frames as observed
        data = {}
        data["video_id"] = [video_ids[0]]
        data["frames"] = frame_list  # 15 observed frames
        data["image"] = self.load_images(video_ids[0], frame_list)  # load iamges
        data["label_speed"] = self.data["driving_speed"][index][
            self.args.observe_length
        ]  # only return 16-th label, 3-class one-hot
        data["label_direction"] = self.data["driving_direction"][index][
            self.args.observe_length
        ]  # only return 16-th label, 3-class one-hot
        data["label_speed_prob"] = self.data["driving_speed_prob"][index][
            self.args.observe_length
        ]  # only return 16-th label, 3-class one-hot
        data["label_direction_prob"] = self.data["driving_direction_prob"][index][
            self.args.observe_length
        ]  # only return 16-th label, 3-class one-hot
        data["description"] = self.data["description"][index][
            self.args.observe_length
        ]  # only return 16-th label, 3-class one-hot

        return data

    def load_images(self, video_name: str, frame_list: list) -> torch.Tensor:
        images = []

        for frame_id in frame_list:
            # load original image
            img_path = os.path.join(
                self.images_path, video_name, str(frame_id).zfill(3) + ".jpg"
            )
            # print(img_path)
            img = self.rgb_loader(img_path)
            # print(img.shape) #1280 x 720
            # Image.fromarray(img).show()
            # img.shape: H x W x C, RGB channel
            # crop pedestrian surrounding image

            if self.transform:
                # print("before transform - img: ", img.shape)
                img = self.transform(img)
                # After transform, changed to tensor, img.shape: C x H x W
            images.append(img)

        return torch.stack(images, dim=0)


class MultiEpochsDataLoader(DataLoader):
    """Custom DataLoader to support caching of the dataset over multiple epochs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
        self._items_to_consume = 0
        self._last_consume_item = -1

    def __len__(self) -> int:
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for _ in range(self._items_to_consume - 1 - self._last_consume_item):
            next(self.iterator)
        self._items_to_consume = len(self)

        for i in range(self._items_to_consume):
            self._last_consume_item = i
            yield next(self.iterator)


class _RepeatSampler:
    """Repeat sampler to support caching of the dataset over multiple epochs."""

    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class YoloDataset(Dataset):
        def __init__(self, dataset_path):
            self.dataset_path = dataset_path
            self.yolo_results = os.listdir(self.dataset_path)

        def __len__(self):
            return len(os.listdir(self.dataset_path))  

        def __getitem__(self, idx):
            result_file = os.path.join(self.dataset_path, self.yolo_results[idx])
            with open(result_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    elements = line.split("\t")
                    vid_id = elements[0]
                    ped_id = elements[1]
                    bbox = ast.literal_eval(elements[2])
                    bbox = torch.tensor(bbox)
                    frames = ast.literal_eval(elements[3])
                    frames = torch.tensor([int(frame) for frame in frames])
            
            data = {
                'video_id': vid_id,
                'ped_id': ped_id,
                'bboxes': bbox,
                'frames': frames
            }
            return data