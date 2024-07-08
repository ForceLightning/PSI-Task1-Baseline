import json
import os
import sys
from datetime import datetime
from test import (get_test_traj_gt, predict_driving, predict_intent,
                  predict_traj, test_intent, validate_intent)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data.prepare_data import get_dataloader, get_dataset
from database.create_database import create_database
from models.build_model import build_model
from opts import get_opts
from torch.utils.tensorboard import SummaryWriter
from train import train_driving, train_intent, train_traj
from ultralytics import YOLO
from utils.evaluate_results import (evaluate_driving, evaluate_intent,
                                    evaluate_traj)
from utils.get_test_intent_gt import get_intent_gt, get_test_driving_gt
from utils.log import RecordResults

from alphapose.preprocess import prep_image
from alphapose.dataloader import crop_from_dets, Mscoco
from alphapose.img import im_to_torch
from alphapose.SPPE.src.main_fast_inference import *
from alphapose.SPPE.src.utils.eval import getPrediction
from alphapose.pPose_nms import pose_nms
from alphapose.visualization import draw_bboxes, vis_frame_fast
from datetime import datetime

# TODO: Integrate tracking functionality using Boxmot
def main(args):

    print("Preparing model...")
    args.source = os.path.join(os.getcwd(), "frames", "video_0131")
    args.inp_dim = 608

    args.inputResH = 320
    args.inputResW = 256
    args.outputResH = 80
    args.outputResW = 64

    yolo_model = YOLO('yolov8s.pt')
    yolo_detections = []

    pose_dataset = Mscoco(args.inputResH, args.inputResW, args.outputResH, args.outputResW)
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    print("Beginning inference...")
    for count, file in enumerate(os.listdir(args.source)):

        start_time = datetime.now()

        img_name = os.path.join(args.source, file)

        _, orig_img_k, _ = prep_image(img_name, args.inp_dim)

        detections = yolo_model.predict(orig_img_k, classes=[0], verbose=False, conf=0.2)
        yolo_dets = {"bboxes": [], "conf": []}

        for box in detections[0].boxes:
            cls = int(box.cls[0].item())
            coords = box.xyxy[0].tolist()
            conf = box.conf[0].item()

            yolo_dets["bboxes"].append(coords)
            yolo_dets["conf"].append([conf])

        yolo_dets["bboxes"] = torch.tensor(yolo_dets["bboxes"])
        yolo_dets["conf"] = torch.tensor(yolo_dets["conf"])

        inps_yolo = torch.zeros(yolo_dets["bboxes"].size(0), 3, args.inputResH, args.inputResW) if yolo_dets["bboxes"] is not None else None
        pt1_yolo = torch.zeros(yolo_dets["bboxes"].size(0), 2) if yolo_dets["bboxes"] is not None else None
        pt2_yolo = torch.zeros(yolo_dets["bboxes"].size(0), 2) if yolo_dets["bboxes"] is not None else None

        inp_yolo = im_to_torch(cv2.cvtColor(orig_img_k, cv2.COLOR_BGR2RGB))
        inps_yolo, pt1_yolo, pt2_yolo = crop_from_dets(inp_yolo, yolo_dets["bboxes"], inps_yolo, pt1_yolo, pt2_yolo, args.inputResH, args.inputResW)

        pose_output = pose_model(inps_yolo.cuda())
        pose_output = pose_output.cpu()
        _, preds_img, preds_scores = getPrediction(
                            pose_output, pt1_yolo, pt2_yolo, args.inputResH, args.inputResW, args.outputResH, args.outputResW)
        
        result = pose_nms(yolo_dets["bboxes"], yolo_dets["conf"], preds_img, preds_scores.detach())
        keypoints = [item["keypoints"] for item in result]

        output_image = vis_frame_fast(orig_img_k, result)
        output_image = draw_bboxes(output_image, yolo_dets["bboxes"])

        end_time = datetime.now()
        print(f"Frame {count+1} inference time: {(end_time - start_time).total_seconds()*1000}ms")

        cv2.imshow(f'frame', output_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break

    else:
        print("Exiting...")

    cv2.destroyAllWindows()
    # writer = SummaryWriter(args.checkpoint_path)
    # recorder = RecordResults(args)
    # ''' 1. Load database '''
    # if not os.path.exists(os.path.join(args.database_path, args.database_file)):
    #     create_database(args)
    # else:
    #     print("Database exists!")
    # train_loader, val_loader, test_loader = get_dataloader(args)
    # val_dataset = get_dataset(args, 'val')    
    # model = YOLO('yolov8s.pt')
    # frames, labels = val_dataset[130]

    sys.exit()

    ''' 2. Create models '''
    model, optimizer, scheduler = build_model(args)
    model = nn.DataParallel(model)

    ''' 3. Train '''
    if args.task_name == 'ped_intent':
        train_intent(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer)
    elif args.task_name == 'ped_traj':
        train_traj(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer)
    elif args.task_name == 'driving_decision':
        train_driving(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer)

    if args.task_name == 'ped_intent':
        # val_gt_file = './test_gt/val_intent_gt.json'
        # if not os.path.exists(val_gt_file):
        #     get_intent_gt(val_loader, val_gt_file, args)
        # predict_intent(model, val_loader, args, dset='val')
        predict_intent(model, yolo_data, args, dset='val')
        # evaluate_intent(val_gt_file, args.checkpoint_path + '/results/val_intent_pred', args)
    # elif args.task_name == 'ped_traj':
    #     val_gt_file = './test_gt/val_traj_gt.json'
    #     if not os.path.exists(val_gt_file):
    #         get_test_traj_gt(model, val_loader, args, dset='val')
    #     predict_traj(model, val_loader, args, dset='val')
    #     score = evaluate_traj(val_gt_file, args.checkpoint_path + '/results/val_traj_pred.json', args)
    # elif args.task_name == 'driving_decision':
    #     val_gt_file = './test_gt/val_driving_gt.json'
    #     if not os.path.exists(val_gt_file):
    #         get_test_driving_gt(model, val_loader, args, dset='val')
    #     predict_driving(model, val_loader, args, dset='val')
    #     score = evaluate_driving(val_gt_file, args.checkpoint_path + '/results/val_driving_pred.json', args)
    #     print("Ranking score of val set: ", score)

    # ''' 4. Test '''
    # test_gt_file = './test_gt/test_intent_gt.json'
    # if not os.path.exists(test_gt_file):
    #     get_intent_gt(test_loader, test_gt_file, args)
    # predict_intent(model, test_loader, args, dset='test')
    # evaluate_intent(test_gt_file, args.checkpoint_path + '/results/test_intent_prediction.json', args)

if __name__ == '__main__':
    # /home/scott/Work/Toyota/PSI_Competition/Dataset
    args = get_opts()

    if args.task_name == 'ped_intent':
        args.database_file = 'intent_database_train.pkl'
        args.intent_model = True

        # intent prediction
        args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
        args.intent_type = 'mean' # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
        args.intent_loss = ['bce']
        args.intent_disagreement = 1  # -1: not use disagreement 1: use disagreement to reweigh samples
        args.intent_positive_weight = 0.5  # Reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)

    # trajectory
    if args.task_name == 'ped_traj':
        args.database_file = 'traj_database_train.pkl'
        args.intent_model = False # if (or not) use intent prediction module to support trajectory prediction
        args.traj_model = True
        args.traj_loss = ['bbox_l1']
        # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
        # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]
    elif args.task_name == 'driving_decision':
        args.database_file = 'driving_database_train.pkl'
        args.driving_loss = ['cross_entropy']

    args.seq_overlap_rate = 0.9 # overlap rate for trian/val set
    args.test_seq_overlap_rate = 1 # overlap for test set. if == 1, means overlap is one frame, following PIE
    args.observe_length = 15
    if args.task_name == 'ped_intent':
        args.predict_length = 1 # only make one intent prediction
    elif args.task_name == 'ped_traj':
        args.predict_length = 45
    elif args.task_name == 'driving_decision':
        args.predict_length = 1  # only make one prediction

    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = 'enlarge'
    args.normalize_bbox = None
   
    # Model
    if args.task_name == 'ped_intent':
        args.model_name = 'lstm_int_bbox'  # LSTM module, with bboxes sequence as input, to predict intent
        args.load_image = False # only bbox sequence as input
    elif args.task_name == 'ped_traj':
        args.model_name = 'lstmed_traj_bbox'
        args.load_image = False # only bbox sequence as input
    elif args.task_name == 'driving_decision':
        args.model_name = 'reslstm_driving_global' 
        args.load_image = True

    if args.load_image:
        args.backbone = 'resnet' # faster-rcnn
        args.freeze_backbone = False
    else:
        args.backbone = None
        args.freeze_backbone = False


    # Train
    args.epochs = 1
    args.batch_size = 128
    if args.task_name == 'ped_intent':
        args.lr = 1e-3
        args.loss_weights = {
            'loss_intent': 1.0,
            'loss_traj': 0.0,
            'loss_driving': 0.0
        }
    elif args.task_name == 'ped_traj':
        args.lr = 1e-2
        args.loss_weights = {
            'loss_intent': 0.0,
            'loss_traj': 1.0,
            'loss_driving': 0.0
        }
    elif args.task_name == 'driving_decision':
        args.lr = 1e-3
        args.loss_weights = {
            'loss_intent': 0.0,
            'loss_traj': 0.0,
            'loss_driving': 1.0
        }

    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    # Record
    now = datetime.now()
    time_folder = now.strftime('%Y%m%d%H%M%S')
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name, args.dataset, args.model_name, time_folder)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    result_path = os.path.join(args.checkpoint_path, 'results')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    main(args)