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


def main(args):
    writer = SummaryWriter(args.checkpoint_path)
    recorder = RecordResults(args)
    ''' 1. Load database '''
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)
    val_dataset = get_dataset(args, 'val')    
    model = YOLO('yolov8s.pt')
    frames, labels = val_dataset[50]

    bboxes = []
    print(frames.shape)
    results = []
    for frame in frames:
        # to plot the image
        # plt.imshow(frame.numpy().transpose(1,2,0))
        # plt.show()
        np_frame = cv2.cvtColor(frame.numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR)
        np_frame = (np_frame * 255).astype(np.uint8)
        result = model.track(np_frame, persist=True, half=True, device='cuda:0', classes=[0])
        # to plot the resultant annotated image
        res = result[0].plot()
        cv2.imshow('frame', res)
        cv2.waitKey(0)
        results.append(result[0])
    for result in results:

        # Print number of bounding boxes
        print("Length:", len(result.boxes))

        # print(result.names)

        for box in result.boxes:
            label = int(box.cls[0].item())
            cords =  box.xyxyn[0].tolist()
            # Only care about the bounding boxes of pedestrian
            if label == 0:
                bboxes.append(cords)
            prob = box.conf[0].item()
            print("Object type :", label)
            print("Coordinates :", cords)
            print("Probability : ", prob)
            print("---")

        # Visualise the predictions
        # img = Image.fromarray(result.plot()[:,:,::-1])

        # print(result.plot().shape)  
        # plt.imshow(img)
        # plt.show()

    yolo_bbox = torch.tensor(bboxes)
    yolo_bbox = torch.unsqueeze(yolo_bbox, dim=0)
    print(yolo_bbox.shape)
    yolo_frames = torch.tensor(labels['frames'])
    yolo_frames = torch.unsqueeze(yolo_frames, dim=0)
    yolo_data = {
        'bboxes': yolo_bbox,
        'frames': yolo_frames,
        'video_id': labels['video_id'], 
        'ped_id': labels['ped_id'],
    }
    sys.exit() # TODO(jiayu): remove this when you are done with YOLO inferencing

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