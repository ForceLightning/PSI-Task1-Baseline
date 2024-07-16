import os
import torch
import json
from torchvision.utils import draw_bounding_boxes
from torchvision.io.image import read_image, write_png
from torchvision.transforms.functional import to_pil_image

class visualisations:

    """ schema 
        # "video_id" : {
        #     "frame_id": {
        #         "boxes" : [], 
        #         "labels" : []
        #         }
        # }
    """
    frame_data = {}

    def __init__(self):
        self.frame_data = {}

    """
        Usage:
            Adds the data to the internal frame_data dictionary
        Params: 
            video_id: string, the video id
            frame_id: int, the frame id
            boxes: tensor, the bounding boxes
            labels: tensor, the labels
        Returns:    
            None
    """
    def add_to_frame_data(self, video_id, frame_id, boxes, labels):
        # print("add_to_frame_data")
        # print(video_id)
        # print(frame_id)
        # print("labels")
        # print(labels)
        
        boxes.unsqueeze_(0)  #convert from (4,) to (1, 4)
        labels = labels.view(-1)  #ensures at least one dimension

        #if video_id does not exist, init the internal dictionary that holds all the frame data to each video
        if video_id not in self.frame_data:
            self.frame_data[video_id] = {}
        #if frame_id does not exist, add the data directly
        if frame_id not in self.frame_data[video_id]:
            self.frame_data[video_id][frame_id] = {'boxes': boxes, 'labels': labels}
            return self.frame_data

        #if frame_id does exist, append the data to the existing data
        self.frame_data[video_id][frame_id]["boxes"] = torch.cat([self.frame_data[video_id][frame_id]["boxes"], boxes])
        self.frame_data[video_id][frame_id]["labels"] = torch.cat([self.frame_data[video_id][frame_id]["labels"], labels])


    """
        Usage:
            Draws bounding boxes on all frames currently in the frame_data dictionary
        Params:
            save_path: string, the path to save the images to
            colors: string, the color of the bounding boxes
            width: int, the width of the bounding boxes
        Returns:
            None
    """
    def draw_all_frame_data_boxes(self, save_path = "frames", colors="red", width=1):
        #draw bounding boxes
        #for each video in frame_data
        #for each frame in video
        #draw bounding boxes

        for video_id in self.frame_data:
            counter = 0
            num_frames = len(self.frame_data[video_id])
            for frame_id in self.frame_data[video_id]:
                #get image
                #set file name to open image
                #DEBUG
                #print(f'video_id: {video_id}, frame_id: {frame_id}')
                fn = f'{save_path}/{video_id}/{str(frame_id).zfill(3)}.jpg'
                try:
                    img = read_image(fn)
                except FileNotFoundError as e:
                    print(f"File not found: {fn}, skipping image")
                    continue
                except OSError as e:
                    print(f"Error reading image {fn}: {e}, skipping image")
                    continue
                    
                #get boxes and labels
                boxes = self.frame_data[video_id][frame_id]["boxes"]
                labels = self.frame_data[video_id][frame_id]["labels"]
                
                #convert the tensor elements to strings
                if labels.numel() > 1:
                    labels_str = [str(value) for value in labels.tolist()]
                else:
                    labels_str = [str(labels[0].item())]
                
                #draw bounding boxes
                box = draw_bounding_boxes(img, 
                                    boxes=boxes, 
                                    labels=labels_str, 
                                    colors=colors, 
                                    width=width)
                
                #DEBUG
                #im = to_pil_image(box.detach())
                #im.show()

                #write to file
                write_fn = fn.split(".",1)[0] #splits string to remove .jpg
                #DEBUG
                #print(f"writing to file, drew {len(labels_str)} boxes, on frame {frame_id} of video {video_id}, file name: {write_fn}")
                #print(write_fn)
                write_png(box, write_fn + "_pred.png")
                counter += 1
                if counter % 10 == 0:
                    print(f"Processed {counter} of {num_frames} frames in video {video_id}.")
            print(f"Done processing video {video_id}, drew {num_frames} frames with bounding boxes.")
        print("Done drawing bounding boxes.")


