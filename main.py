from opts import get_opts
from datetime import datetime
import os
import json
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.prepare_data import get_dataloader, get_video_dimensions, \
consolidate_yolo_data, save_data_to_txt, visualise_annotations, visualise_intent
from database.create_database import create_database
from models.build_model import build_model
from train import train_intent
from test import validate_intent, test_intent, predict_intent
from utils.log import RecordResults
from utils.evaluate_results import evaluate_intent
from utils.get_test_intent_gt import get_intent_gt
from sklearn.model_selection import ParameterSampler
import numpy as np
import glob
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

from boxmot import DeepOCSORT
from data.custom_dataset import YoloDataset

def main(args):
    # writer = SummaryWriter(args.checkpoint_path)
    # recorder = RecordResults(args)
    # ''' 1. Load database '''
    # if not os.path.exists(os.path.join(args.database_path, args.database_file)):
    #     create_database(args)
    # else:
    #     print("Database exists!")
    # train_loader, val_loader, test_loader = get_dataloader(args)
    
    # Set args.classes to 0 for pedestrian tracking
    args.classes = 0

    # If video source source is from test
    args.source = os.path.join(os.getcwd(), "PSI2.0_Test", "videos", "video_0147.mp4")
    
    # If video source is from val
    # args.source = os.path.join(os.getcwd(), "PSI_Videos", "videos", "video_0111.mp4")
    file_name = args.source.split("\\")[-1].split(".")[0]
    
    model = YOLO("yolov8s.pt")
    tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=False,
    )
    
    # This tracking process can be functionised when got time
    vid = cv2.VideoCapture(args.source)

    frame_number = 0
    while True:
        ret, im = vid.read()

        if not ret :
            break
        
        # Increment frame number
        frame_number += 1

        result = model.predict(im, classes=[0], verbose=False)

        dets = []
        result = result[0]
        
        for box in result.boxes:
            cls = int(box.cls[0].item())
            cords = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            id = box.id
            dets.append([cords[0], cords[1], cords[2], cords[3], conf, cls])
                
        dets = np.array(dets)

        if len(dets) == 0:
            dets = np.empty((0, 6))
            
        tracking_results = tracker.update(dets, im) 
        if len(tracking_results) > 0:
            x1 = tracking_results[0][0]
            y1 = tracking_results[0][1]
            x2 = tracking_results[0][2]
            y2 = tracking_results[0][3]
            id = tracking_results[0][4]
            conf = tracking_results[0][5]
            cls = tracking_results[0][6]
            with open(file_name + ".txt", 'a') as f:
                f.write(f"{int(id)} {x1} {y1} {x2} {y2} {conf} {int(cls)} {frame_number}\n")
                
        tracker.plot_results(im, show_trajectories=False)
        
        # break on pressing q or space
    #     cv2.imshow('BoxMOT detection', im)     
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord(' ') or key == ord('q'):
    #         break

    # vid.release()
    # cv2.destroyAllWindows()

    bbox_holder, frames_holder, video_id = consolidate_yolo_data(file_name)
    save_data_to_txt(bbox_holder, frames_holder, video_id)    

    example_data = YoloDataset(os.path.join(os.getcwd(), "yolo_results_data"))

    example_loader = torch.utils.data.DataLoader(example_data, batch_size=args.batch_size, shuffle=False,
                                           pin_memory=True, sampler=None, num_workers=4)

    ''' 2. Create models '''
    # model, optimizer, scheduler = build_model(args)
    # model = nn.DataParallel(model)

    ''' 3. Train '''
    # train_intent(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer)

    # Load Pretrained model
    args.tcn_kernel_size = 4
    args.kernel_size = 8
    
    model, _, _ = build_model(args)
    model = nn.DataParallel(model)
    # Assuming pth file is at the root directory for now
    model.load_state_dict(torch.load('latest.pth'))

    # val_gt_file = './test_gt/val_intent_gt.json'
    # if not os.path.exists(val_gt_file):
    #     get_intent_gt(val_loader, val_gt_file, args)
    
    # Set dset to test to write results to test_gt folder for now
    predict_intent(model, example_loader, args, dset='test')
    # Visualise specific bbox from specific frame fed into TCN for sanity check
    # visualise_annotations(os.path.join(os.getcwd(), "yolo_results_data", "1.txt"), 0)
    visualise_intent(os.path.join(os.getcwd(), file_name + ".txt"),
                    os.path.join(os.getcwd(), "test_gt", "test_intent_pred"))

    # val_accuracy = evaluate_intent(val_gt_file, args.checkpoint_path + '/results/val_intent_pred', args)

    ''' 4. Test '''
    # test_accuracy = 0.0
    # if test_loader is not None:
    #     test_gt_file = './test_gt/test_intent_gt.json'
    #     if not os.path.exists(test_gt_file):
    #         get_intent_gt(test_loader, test_gt_file, args)
    #     predict_intent(model, test_loader, args, dset='test')
    #     test_accuracy = evaluate_intent(test_gt_file, args.checkpoint_path + '/results/test_intent_pred', args)
    
    return 

if __name__ == '__main__':
    args = get_opts()
    # Task
    args.task_name = 'ped_intent'
    args.persist_dataloader = True

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

    args.seq_overlap_rate = 0.9 # overlap rate for trian/val set
    args.test_seq_overlap_rate = 1 # overlap for test set. if == 1, means overlap is one frame, following PIE
    args.observe_length = 15
    if args.task_name == 'ped_intent':
        args.predict_length = 1 # only make one intent prediction
    elif args.task_name == 'ped_traj':
        args.predict_length = 45

    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = 'enlarge'
    args.normalize_bbox = None
    # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
    # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]

    # Model
    args.model_name = 'tcn_int_bbox'  # TCN module, with bboxes sequence as input, to predict intent
    args.load_image = False # only bbox sequence as input
    if args.load_image:
        args.backbone = 'resnet'
        args.freeze_backbone = False
    else:
        args.backbone = None
        args.freeze_backbone = False

    # Train
    hyperparameter_list = {
        'lr': [1e-4,3e-4,1e-3,3e-3],
        'batch_size': [64,128,256,512,1024],
        'epochs': [50],
        'n_layers': [2,4,8],
        'kernel_size': [2,4,8],
    }

    n_random_samples = 5

    parameter_samples = list(ParameterSampler(hyperparameter_list, n_iter=n_random_samples))

    args.loss_weights = {
        'loss_intent': 1.0,
        'loss_traj': 0.0,
        'loss_driving': 0.0
    }
    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    best_val_accuracy = 0.0
    best_hyperparameters = None

    checkpoint_path = args.checkpoint_path

    # for params in parameter_samples:
    #     args.lr = params['lr']
    #     args.batch_size = params['batch_size']
    #     args.epochs = params['epochs']
    #     args.kernel_size = params['kernel_size']
    #     args.n_layers = params['n_layers']

    #     # Record
    #     now = datetime.now()
    #     time_folder = now.strftime('%Y%m%d%H%M%S')
    #     args.checkpoint_path = os.path.join(checkpoint_path, args.task_name, args.dataset, args.model_name,
    #                                         time_folder)
    #     if not os.path.exists(args.checkpoint_path):
    #         os.makedirs(args.checkpoint_path)
    #     with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
    #         json.dump(args.__dict__, f, indent=4)

    #     result_path = os.path.join(args.checkpoint_path, 'results')
    #     if not os.path.isdir(result_path):
    #         os.makedirs(result_path)

    #     print("Running with Parameters:", params)  # Print the current parameters
    #     val_accuracy, test_accuracy = main(args)
    #     print("Validation Accuracy:", val_accuracy)
    #     print("Test Accuracy:", test_accuracy)

    #     if val_accuracy > best_val_accuracy:
    #         best_val_accuracy = val_accuracy
    #         best_hyperparameters = params

    # print("Best Validation Accuracy:", best_val_accuracy)
    # print("Best Hyperparameters:", best_hyperparameters)
    args.n_layers = 4
    main(args)