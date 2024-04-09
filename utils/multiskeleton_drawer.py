import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

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
# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

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
detected_keypoints = []

class multiskeleton_drawer:
    def __init__(self) -> None:
        pass

    #def draw_skeleton(self, frame, skeleton, color=(0, 255, 0), width=2):

    def pose_estimation(self, frame):


        #create a blob from the frame
        #blob = cv.dnn.blobFromImage(frame, 1.0, (frame_width, frame_height), (127.5, 127.5, 127.5), swapRB=True, crop=False) #this was from single skeleton
        blob = cv.dnn.blobFromImage(frame, 1.0/ 255, (frame_width, frame_height), (0,0,0), swapRB=False, crop=False)

        #set the input
        mobile_net.setInput(blob)

        #get the output
        output = mobile_net.forward()

        #get the points of the body parts
        H = output.shape[2]
        W = output.shape[3]

        print("Output Shape : ", output.shape) #DEBUG

        prob_map = output[0, 0, :, :]
        prob_map = cv.resize(prob_map, (frame_width, frame_height))

        plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) #DEBUG
        plt.imshow(prob_map, alpha = 0.6) #DEBUG
        plt.show() #DEBUG
        #exit() #DEBUG

        map_smooth = cv.GaussianBlur(prob_map, (3, 3), 0, 0)
        mapmask = np.uint8(map_smooth > threshold)
        
        #find blobs
        _, contours, _ = cv.findContours(mapmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        #detected_keypoints = []
        #for each blob find maxima
        for contour in contours:
            blob_mask = np.zeros(mapmask.shape)
            blob_mask = cv.fillConvexPoly(blob_mask, contour, 1)
            masked_prob_map = map_smooth * blob_mask
            _, max_val, _, max_loc = cv.minMaxLoc(masked_prob_map)
            detected_keypoints.append(max_loc+ (prob_map[max_loc[1], max_loc[0]],))
            
            
        #pafA = output[0, 19, :, :]
        
        exit()
        #empty list to store the points

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
    
# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv.resize(pafA, (frame_width, frame_height))
        pafB = cv.resize(pafB, (frame_width, frame_height))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



#main
if __name__ == "__main__":
    skel = multiskeleton_drawer()

    #load the pre-trained model
    #mobile_net = cv.dnn.readNetFromTensorflow("PSI-Intent-Prediction/utils/graph_opt.pb")
    protoFile = "PSI-Intent-Prediction/models/vis_models/pose_deploy_linevec.prototxt"
    weightsFile = "PSI-Intent-Prediction/models/vis_models/pose_iter_440000.caffemodel"
    mobile_net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

    #inWidth = 368 #read widths and heights from images???
    #inHeight = 368 #???

    #threshold for the confidence score
    threshold = 0.1 

    #test image
    img = cv.imread(f"frames/video_0016/227.jpg")

    cropped_img = img [300: 500, 400: 600]
    #plt.imshow(cropped_img)
    #plt.show()
    #exit()
    #show image DEBUG
    #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #plt.show()
    #get the height and width of the frame
    frame_width = cropped_img.shape[1]
    frame_height = cropped_img.shape[0]
    points = skel.pose_estimation(cropped_img)
    #skeleton = skel.draw_skeleton(img, points)

    #show image DEBUG
    #plt.imshow(cv.cvtColor(skeleton, cv.COLOR_BGR2RGB))
    #plt.show()


