import os
from typing import Any, Literal, TypedDict
from dataclasses import dataclass, field

ROOT_DIR = "../"


@dataclass
class DefaultArguments:
    """Default arguments for training.

    :param dataset: The dataset to perform train/val/test on.
    :type dataset: Literal['PSI2.0'] | Literal['PSI1.0']
    :param task_name: The task to perform train/val/test on.
    :type task_name: Literal['ped_intent'] | Literal['ped_traj'] | Literal['driving_decision']
    :param video_splits: Path to the train/val/test splitting JSON.
    :type video_splits: os.PathLike | str
    :param dataset_root_path: Path to the dataset root (dataset must be a directory in this path).
    :type dataset_root_path: os.PathLike | str
    :param database_path: Path to the database directory.
    :type database_path: os.PathLike | str
    :param database_file: Filename of pickled database.
    :type database_file: str
    :param fps: FPS of original video, PSI and PEI == 30.
    :type fps: int
    :param seq_overlap_rate: Train/Val rate of the overlapping frames of sliding windows,
        `(1 - rate) * seq_length` is the step size.
    :type seq_overlap_rate: float
    :param test_seq_overlap_rate: Test overlap rate of the overlapping frames of sliding windows,
        `(1 - rate) * seq_length` is the step size.
    :type test_seq_overlap_rate: float
    :param intent_num: Type of intention categories. 2: {cross / not-cross} | 3 {not-sure}
    :type intent_num: Literal[2] | Literal[3]
    :param intent_type: Type of intention lables, out of 24 annotators. Only when set to separate
        can the NLP reasoning be of use, otherwise you may take the weighted mean of NLP
        embeddings, defaults to 'mean'.
    :type intent_type: Literal['major'] | Literal['mean'] | Literal['separate'] | Literal['soft_vote']
    :param observe_length: Sequence length of one observed clip, defaults to 15.
    :type observe_length: int
    :param predict_length: Sequence length of predicted trajectory/intention, defaults to 45.
    :type predict_length: int
    :param max_track_size: Sequence length of observed + predicted trajectory/intention, defaults to 60.
    :type max_track_size: int
    :param crop_mode: Cropping mode of cropping the region surrounding the pedestrian, defaults to 'enlarge'.
    :type crop_mode: str
    :param balance_data: Whether to use a balanced data sampler with random class weights, defaults to False.
    :type balance_data: bool
    :param normalize_bbox: How to normalize bounding boxes, defaults to None.
    :type normalize_bbox: Literal['L2'] | Literal['subtract_first_frame'] | Literal['divide_image_size'] | None
    :param image_shape: Image shape: PSI (1280, 720), defaults to (1280, 720).
    :type image_shape: tuple[int, int]
    :param load_image: Whether to load the image to the backbone model, defaults to False.
    :type load_image: bool

    :param backbone: Backbone model type, defaults to ''.
    :type backbone: Literal['resnet50'] | Literal['vgg16'] | Literal['faster_rcnn']
    :param freeze_backbone: Whether to freeze the backbone model layers during training, defaults to False.
    :type freeze_backbone: bool
    :param model_name: Model name, defaults to 'lstm'.
    :type model_name: str
    :param intent_model: Whether to use an intent model, defaults to True.
    :type intent_model: bool
    :param traj_model: Whether to use a trajectory model, defaults to False.
    :type traj_model: bool
    :param model_configs: Framework information, defaults to {}.
    :type model_configs: dict

    :param checkpoint_path: Path to save model checkpoints, defaults to './ckpts'.
    :type checkpoint_path: str
    :param epochs: Total number of training epochs, defaults to 1000.
    :type epochs: int
    :param batch_size: Batch size for training, defaults to 128.
    :type batch_size: int
    :param lr: Learning rate for training, defaults to 1e-3.
    :type lr: float
    :param resume: Checkpoint path + filename to be resumed, defaults to ''.
    :type resume: str
    :param loss_weights: Weight of loss terms {loss_intent, loss_traj}, defaults to {}.
    :type loss_weights: dict[str, float]
    :param intent_loss: Intent loss function, defaults to 'bce'.
    :type intent_loss: Literal['bce'] | Literal['mse'] | Literal['cross_entropy']
    :param intent_disagreement: Disagreement threshold for intent, defaults to -1.0.
    :type intent_disagreement: float
    :param ignore_uncertain: Whether to ignore uncertain data, defaults to False.
    :type ignore_uncertain: bool
    :param intent_positive_weight: Weight for positive intent, defaults to 1.0.
    :type intent_positive_weight: float
    :param traj_loss: Loss function for trajectory, defaults to ['mse'].
    :type traj_loss: list[str]
    :param steps_per_epoch: Number of mini-batches per epoch, used for the OneCycleLR, defaults to 0.
    :type steps_per_epoch: int

    :param val_freq: Validation frequency, defaults to 10.
    :type val_freq: int
    :param test_freq: Test frequency, defaults to 10.
    :type test_freq: int
    :param print_freq: Print frequency, defaults to 10.
    :type print_freq: int
    :param persist_dataloader: Whether to persist the dataloader, defaults to True.
    :type persist_dataloader: bool
    :param n_layers: Number of layers, defaults to 4.
    :type n_layers: int
    :param kernel_size: Kernel size for TCN, defaults to 4.
    :type kernel_size: int
    """

    dataset: Literal["PSI2.0"] | Literal["PSI1.0"] = "PSI2.0"
    task_name: (
        Literal["ped_intent"] | Literal["ped_traj"] | Literal["driving_decision"]
    ) = "ped_intent"
    video_splits: os.PathLike[Any] | str = os.path.join(
        ROOT_DIR, "splits", "PSI2_split.json"
    )
    dataset_root_path: os.PathLike[Any] | str = os.path.abspath(ROOT_DIR)
    database_path: os.PathLike[Any] | str = os.path.join(ROOT_DIR, "database")
    database_file: str = "intent_database_train.pkl"
    fps: int = 30
    seq_overlap_rate: float = 0.9
    test_seq_overlap_rate: float = 1.0
    intent_num: Literal[2] | Literal[3] = 2
    intent_type: (
        Literal["major"] | Literal["mean"] | Literal["separate"] | Literal["soft_vote"]
    ) = "mean"
    observe_length: int = 15
    predict_length: int = 45
    max_track_size: int = 60
    crop_mode: str = "enlarge"
    balance_data: bool = False
    normalize_bbox: (
        Literal["L2"]
        | Literal["subtract_first_frame"]
        | Literal["divide_image_size"]
        | None
    ) = None
    image_shape: tuple[int, int] = (1280, 720)
    load_image: bool = False

    # About models
    backbone: (
        Literal[""] | Literal["resnet50"] | Literal["vgg16"] | Literal["faster_rcnn"]
    ) = ""
    freeze_backbone: bool = False
    model_name: str = "lstm"
    intent_model: bool = True
    traj_model: bool = False
    model_configs: dict = field(default_factory=dict)

    # About training
    checkpoint_path: str = "./ckpts"
    epochs: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    resume: str = ""  # ckpt path + filename to be resumed
    loss_weights: dict[str, float] = field(
        default_factory=dict
    )  # weight of loss terms {loss_intent, loss_traj}
    intent_loss: Literal["bce"] | Literal["mse"] | Literal["cross_entropy"] = "bce"
    intent_disagreement: float = -1.0
    ignore_uncertain: bool = False
    intent_positive_weight: float = 1.0
    traj_loss: list[str] = field(default_factory=lambda: ["mse"])
    steps_per_epoch: int = 0

    # Other parameters
    val_freq: int = 10
    test_freq: int = 10
    print_freq: int = 10
    persist_dataloader: bool = True
    n_layers: int = 4
    kernel_size: int = 4


class ModelOpts(TypedDict):
    enc_in_dim: int
    enc_out_dim: int
    dec_in_emb_dim: int | None
    dec_out_dim: int
    output_dim: int
    n_layers: int
    dropout: float
    kernel_size: int
    observe_length: int
    predict_length: int
    return_sequence: bool
    output_activation: (
        Literal["None"]
        | Literal["tanh"]
        | Literal["sigmoid"]
        | Literal["softmax"]
        | None
    )
    use_skip_connections: bool
    num_heads: int
