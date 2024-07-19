import argparse
from pathlib import Path


def init_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch implementation of the PSI2.0")
    # about data
    _ = parser.add_argument(
        "--dataset", type=str, default="PSI2.0", help="task name: [PSI1.0 | PSI2.0]"
    )
    _ = parser.add_argument(
        "--task_name",
        type=str,
        default="ped_intent",
        help="task name: [ped_intent | ped_traj | driving_decision]",
    )
    _ = parser.add_argument(
        "--video_splits",
        type=str,
        default="./PSI2.0_TrainVal/splits/PSI2_split.json",
        help="video splits, [PSI100_split | PSI200_split | PSI200_split_paper]",
    )
    _ = parser.add_argument(
        "--dataset_root_path",
        type=str,
        default="./",
        help="Path of the dataset, e.g. frames/video_0001/000.jpg",
    )
    _ = parser.add_argument(
        "--database_path",
        type=str,
        default="./database",
        help="Path of the database created based on the cv_annotations and nlp_annotations",
    )
    _ = parser.add_argument(
        "--database_file",
        type=str,
        default="intent_database_train.pkl",
        help="Filename of the database created based on the cv_annotations and nlp_annotations",
    )
    _ = parser.add_argument(
        "--fps", type=int, default=30, help=" fps of original video, PSI and PEI == 30."
    )
    _ = parser.add_argument(
        "--seq_overlap_rate",
        type=float,
        default=0.9,  # 1 means every stride is 1 frame
        help="Train/Val rate of the overlap frames of slideing windown, (1-rate)*seq_length is the step size",
    )
    _ = parser.add_argument(
        "--test_seq_overlap_rate",
        type=float,
        default=1,  # 1 means every stride is 1 frame
        help="Test overlap rate of the overlap frames of slideing windown, (1-rate)*seq_length is the step size",
    )
    _ = parser.add_argument(
        "--intent_num",
        type=int,
        default=2,
        help="Type of intention categories. [2: {cross/not-cross} | 3 {not-sure}]",
    )
    _ = parser.add_argument(
        "--intent_type",
        type=str,
        default="mean",
        help="Type of intention labels, out of 24 annotators. [major | mean | separate | soft_vote];"
        "only when separate, the nlp reasoning can help, otherwise may take weighted mean of the nlp embeddings",
    )
    _ = parser.add_argument(
        "--observe_length",
        type=float,
        default=15,
        help="Sequence length of one observed clips",
    )
    _ = parser.add_argument(
        "--predict_length",
        type=float,
        default=45,
        help="Sequence length of predicted trajectory/intention",
    )
    _ = parser.add_argument(
        "--max_track_size",
        type=float,
        default=60,
        help="Sequence length of observed + predicted trajectory/intention",
    )
    _ = parser.add_argument(
        "--crop_mode",
        type=str,
        default="enlarge",
        help="Cropping mode of cropping the pedestrian surrounding area",
    )
    _ = parser.add_argument(
        "--balance_data",
        type=bool,
        default=False,
        help="Balance data sampler with randomly class-wise weighted",
    )
    _ = parser.add_argument(
        "--normalize_bbox",
        type=str,
        default=None,
        help="If normalize bbox. [L2 | subtract_first_frame | divide_image_size]",
    )
    _ = parser.add_argument(
        "--image_shape",
        type=tuple,
        default=(1280, 720),
        help="Image shape: PSI(1280, 720).",
    )
    _ = parser.add_argument(
        "--load_image",
        type=bool,
        default=False,
        help="Do not load image to backbone if not necessary",
    )

    # about models
    _ = parser.add_argument(
        "--backbone",
        type=str,
        default="",
        help="Backbone type [resnet50 | vgg16 | faster_rcnn]",
    )
    _ = parser.add_argument(
        "--freeze_backbone", type=bool, default=False, help="[True | False]"
    )
    _ = parser.add_argument(
        "--model_name", type=str, default="lstm", help="model name, [lstm, lstmed]"
    )
    _ = parser.add_argument(
        "--intent_model", type=bool, default=True, help="[True | False]"
    )
    _ = parser.add_argument(
        "--traj_model", type=bool, default=False, help="[True | False]"
    )
    _ = parser.add_argument(
        "--model_configs", type=dict, default={}, help="framework information"
    )

    # about training
    _ = parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="ckpts",
        help="Path of the stored checkpoints",
    )
    _ = parser.add_argument(
        "--epochs", type=int, default=1000, help="Total number of training epochs"
    )
    _ = parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size of dataloader"
    )
    _ = parser.add_argument(
        "--lr", type=float, default=1e-3, help="General learning rate, default as 1e-3"
    )
    _ = parser.add_argument(
        "--resume", type=str, default="", help="ckpt path+filename to be resumed."
    )
    _ = parser.add_argument(
        "--loss_weights",
        type=dict,
        default={},
        help="weights of loss terms, {loss_intent, loss_traj}",
    )
    _ = parser.add_argument(
        "--intent_loss",
        type=list,
        default=["bce"],
        help="loss for intent output. [bce | mse | cross_entropy]",
    )
    _ = parser.add_argument(
        "--intent_disagreement",
        type=float,
        default=-1.0,
        help="weather use disagreement to reweight intent loss.threshold to filter training data."
        + "consensus > 0.5 are selected and reweigh loss; -1.0 means not use; 0.0, means all are used.",
    )
    _ = parser.add_argument(
        "--ignore_uncertain",
        type=bool,
        default=False,
        help="ignore uncertain training samples, based on intent_disagreement",
    )
    _ = parser.add_argument(
        "--intent_positive_weight",
        type=float,
        default=1.0,
        help="weight for intent bce loss: e.g., 0.5 ~= n_neg_class_samples(5118)/n_pos_class_samples(11285)",
    )
    _ = parser.add_argument(
        "--traj_loss",
        type=list,
        default=["mse"],
        help="loss for intent output. [bce | mse | cross_entropy]",
    )

    # other parameteres
    _ = parser.add_argument(
        "--val_freq", type=int, default=10, help="frequency of validate"
    )
    _ = parser.add_argument(
        "--test_freq", type=int, default=10, help="frequency of test"
    )
    _ = parser.add_argument(
        "--print_freq", type=int, default=10, help="frequency of print"
    )
    _ = parser.add_argument(
        "--persist_dataloader",
        type=bool,
        default=True,
        help="persistent_workers in multi epochs dataloader",
    )
    _ = parser.add_argument(
        "--n_layers", type=int, default=4, help="number of layers in the model"
    )
    _ = parser.add_argument(
        "--kernel_size", type=int, default=4, help="kernel size of TCN"
    )

    _ = parser.add_argument(
        "--comment", type=str, default="", help="Description of experiment"
    )
    parser.add_argument(
        "--source", type=str, default="0", help="file/dir/URL/glob, 0 for webcam"
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="intersection over union (IoU) threshold for NMS",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--show", action="store_true", help="display tracking video results"
    )
    parser.add_argument(
        "--save", action="store_true", help="save video tracking results"
    )
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    # parser.add_argument(
    #     "--classes",
    #     nargs="+",
    #     type=int,
    #     help="filter by class: --classes 0, or --classes 0 2 3",
    # )
    # parser.add_argument(
    #     "--project",
    #     default=ROOT / "runs" / "track",
    #     help="save results to project/name",
    # )
    # parser.add_argument("--name", default="exp", help="save results to project/name")
    # parser.add_argument(
    #     "--exist-ok",
    #     action="store_true",
    #     help="existing project/name ok, do not increment",
    # )
    # parser.add_argument(
    #     "--half", action="store_true", help="use FP16 half-precision inference"
    # )
    # parser.add_argument(
    #     "--vid-stride", type=int, default=1, help="video frame-rate stride"
    # )
    # parser.add_argument(
    #     "--show-labels", action="store_false", help="either show all or only bboxes"
    # )
    # parser.add_argument(
    #     "--show-conf", action="store_false", help="hide confidences when show"
    # )
    # parser.add_argument(
    #     "--show-trajectories", action="store_true", help="show confidences"
    # )
    # parser.add_argument(
    #     "--save-txt", action="store_false", help="save tracking results in a txt file"
    # )
    # parser.add_argument(
    #     "--save-id-crops",
    #     action="store_true",
    #     help="save each crop to its respective id folder",
    # )
    # parser.add_argument(
    #     "--line-width",
    #     default=None,
    #     type=int,
    #     help="The line width of the bounding boxes. If None, it is scaled to the image size.",
    # )
    # parser.add_argument(
    #     "--per-class",
    #     default=False,
    #     action="store_true",
    #     help="not mix up classes when tracking",
    # )
    # parser.add_argument(
    #     "--verbose", default=True, action="store_true", help="print results per frame"
    # )
    # parser.add_argument(
    #     "--agnostic-nms", default=False, action="store_true", help="class-agnostic NMS"
    # )

    _ = parser.add_argument(
        "--profile_execution", action="store_true", help="Record execution metrics"
    )

    _ = parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Compile the model with torch.compile",
    )

    ### Pipeline specific arguments ###
    _ = parser.add_argument(
        "--yolo_pipeline_weights",
        type=str,
        default="yolov8s-pose.pt",
        help="Path to the weights of the YOLO pipeline model.",
    )

    _ = parser.add_argument(
        "--boxmot_tracker_weights",
        type=str,
        default="osnet_x0_25_msmt17.pt",
        help="Path to the weights of the BoxMoT tracker.",
    )

    _ = parser.add_argument(
        "--tracker",
        "-t",
        type=str,
        default="deepocsort",
        choices=["botsort", "byte", "deepocsort", "hrnet", "demo"],
        help="The tracker to use for tracking.",
    )

    _ = parser.add_argument(
        "--hrnet_yolo_ver",
        type=str,
        default="v8",
        choices=["v3", "v5", "v8"],
        help="The version of the HRNet-YOLO model to use.",
    )

    return parser


def get_opts() -> argparse.Namespace:
    """
    Gets the command line arguments and returns the parsed arguments.

    See `utils.args.DefaultArguments` for more information.

    Returns:
        argparse.Namespace

    Example:
        >>> # get arguments
        >>> args = get_opts()
    """

    parser = init_args()

    return parser.parse_args()
