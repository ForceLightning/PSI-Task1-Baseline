import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#individual body parts and their respective index
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

#pose pairs, each pair is a line that connects two body parts
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

SKELETON_COLORS = [
    (255, 0, 0),    # Red
    (255, 64, 0),
    (255, 128, 0),
    (255, 191, 0),
    (255, 255, 0),   # Yellow
    (191, 255, 0),
    (128, 255, 0),
    (64, 255, 0),
    (0, 255, 0),     # Green
    (0, 255, 64),
    (0, 255, 128),
    (0, 255, 191),
    (0, 255, 255),   # Cyan
    (0, 191, 255),
    (0, 128, 255),
    (0, 64, 255),
    (0, 0, 255),     # Blue
    (64, 0, 255),
    (128, 0, 255),
    (191, 0, 255),
    (255, 0, 255)    # Purple
]

class skeleton_drawer:
    def __init__(self) -> None:
        pass

    #def draw_skeleton(self, frame, skeleton, color=(0, 255, 0), width=2):

    def pose_estimation(self, frame):
        #get the height and width of the frame
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        #create a blob from the frame
        blob = cv.dnn.blobFromImage(frame, 1.0, (frame_width, frame_height), (127.5, 127.5, 127.5), swapRB=True, crop=False)

        #set the input
        mobile_net.setInput(blob)

        #get the output
        output = mobile_net.forward()

        #get the points of the body parts
        H = output.shape[2]
        W = output.shape[3]

        print(output.shape)

        #empty list to store the points
        points = []

        #for each body part, get the confidence map and find the global maxima
        #only needs first 19 body parts
        for i in range(19):
            #get the confidence map of the body part
            prob_map = output[0, i, :, :]

            #find the global maxima of the prob_map
            min_val, prob, min_loc, point = cv.minMaxLoc(prob_map)

            #scale the point to the size of the frame
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H

            #if the prob is greater than the threshold, add the point to the list
            if prob > threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        return points
    
    def draw_skeleton(self, frame, skeleton, color=(255, 0, 0), width=2):
        #draw the lines between the body parts
        for index, pair in enumerate(POSE_PAIRS):
            part_a = pair[0]
            part_b = pair[1]

            #get the index of the body parts
            index_a = BODY_PARTS[part_a]
            index_b = BODY_PARTS[part_b]

            #if the index is not None, draw the line
            if skeleton[index_a] and skeleton[index_b]:
                cv.line(frame, skeleton[index_a], skeleton[index_b], SKELETON_COLORS[index], width)
                cv.ellipse(frame, skeleton[index_a], (width, width), 0, 0, 360, color, cv.FILLED)
                cv.ellipse(frame, skeleton[index_b], (width, width), 0, 0, 360, color, cv.FILLED)

        return frame




if __name__ == "__main__":
    #load the pre-trained model
    mobile_net = cv.dnn.readNetFromTensorflow("PSI-Intent-Prediction/models/vis_models/graph_opt.pb")

    #inWidth = 368 #read widths and heights from images???
    #inHeight = 368 #???

    #threshold for the confidence score
    threshold = 0.1 

    skel = skeleton_drawer()

    starting_frame = 90
    #repeat for 60 frames
    for i in range(60):
        
        #read the image
        img = cv.imread(f"frames/video_0010/{str(starting_frame + i).zfill(3)}.jpg")

        
        #get the points of the body parts
        points = skel.pose_estimation(img)

        #draw the skeleton
        skeleton = skel.draw_skeleton(img, points)

        #save the image
        cv.imwrite(f"outputs/video_0010_skeletons/{str(starting_frame + i).zfill(3)}.jpg", cv.cvtColor(skeleton, cv.COLOR_BGR2RGB))
        print("writing frame", starting_frame + i)
    
    print("done")
    #test image
    # img = cv.imread(f"frames/video_0016/227.jpg")

    # #show image DEBUG
    # #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # #plt.show()


    # points = skel.pose_estimation(img)
    # skeleton = skel.draw_skeleton(img, points)

    # #show image DEBUG
    # plt.imshow(cv.cvtColor(skeleton, cv.COLOR_BGR2RGB))
    # plt.show()
