import cv2
import time
import numpy as np
from random import randint
import argparse
import matplotlib.pyplot as plt

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

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
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

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

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


parser = argparse.ArgumentParser(description='Run keypoint detection') 
parser.add_argument("--device", default="cpu", help="Device to inference on"  )
parser.add_argument("--image_file", default="group.jpg", help="Input image")

args = parser.parse_args()



#image1 = image[300: 500, 400: 600]

protoFile = "PSI-Intent-Prediction/models/vis_models/pose_deploy_linevec.prototxt"
weightsFile = "PSI-Intent-Prediction/models/vis_models/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]




# image1 = cv2.imread(f"test_image.jpg")

# frameWidth = image1.shape[1]
# frameHeight = image1.shape[0]

# # Fix the input Height and get the width according to the Aspect Ratio
# inHeight = 368
# inWidth = int((inHeight/frameHeight)*frameWidth)

# inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
#                           (0, 0, 0), swapRB=False, crop=False)

# net.setInput(inpBlob)
# output = net.forward()
# print("Time Taken in forward pass = {}".format(time.time() - t))

# detected_keypoints = []
# keypoints_list = np.zeros((0,3))
# keypoint_id = 0
# threshold = 0.1

# for part in range(nPoints):
#     probMap = output[0,part,:,:]
#     probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
#     keypoints = getKeypoints(probMap, threshold)
#     print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
#     keypoints_with_id = []
#     for i in range(len(keypoints)):
#         keypoints_with_id.append(keypoints[i] + (keypoint_id,))
#         keypoints_list = np.vstack([keypoints_list, keypoints[i]])
#         keypoint_id += 1

#     detected_keypoints.append(keypoints_with_id)


# frameClone = image1.copy()
# for i in range(nPoints):
#     for j in range(len(detected_keypoints[i])):
#         cv2.circle(frameClone, detected_keypoints[i][j][0:2], 2, colors[i], -1, cv2.LINE_AA)
# cv2.imshow("Keypoints",frameClone)


# valid_pairs, invalid_pairs = getValidPairs(output)
# personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

# for i in range(17):
#     for n in range(len(personwiseKeypoints)):
#         index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
#         if -1 in index:
#             continue
#         B = np.int32(keypoints_list[index.astype(int), 0])
#         A = np.int32(keypoints_list[index.astype(int), 1])
#         cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 1, cv2.LINE_AA)


# cv2.imshow("Detected Pose" , frameClone)
# cv2.waitKey(0)

num_frames = 30
starting_frame = 120
file_path = "frames/video_0131"
output_path = "outputs/video_0131"
output_file = "outputs/vid0013_169_cropping_pred.png"

#file = "frames/video_0013/169.jpg"
#output_file = "outputs/vid0013_169_pred.png"
#draw for 30 frames
for counter in range(num_frames):
    image1 = cv2.imread(f"{file_path}/{str(starting_frame + counter).zfill(3)}.jpg")
    output_file = f"{output_path}/{str(starting_frame + counter).zfill(3)}_skeleton.png"
    #file = "frames/video_0013/169.jpg"
    #image1 = cv2.imread(file)

    """    #crop image
    
    xMax = 0
    yMax = 0
    xMin = 99999
    yMin = 99999
    
    bboxes = [
                    [
                        688.33,
                        308.21,
                        701.07,
                        336.26
                    ],
                    [
                        688.68,
                        308.21,
                        701.06,
                        336.26
                    ],
                    [
                        689.04,
                        308.21,
                        701.04,
                        336.26
                    ],
                    [
                        689.39,
                        308.21,
                        701.03,
                        336.26
                    ],
                    [
                        689.75,
                        308.21,
                        701.01,
                        336.26
                    ],
                    [
                        690.1,
                        308.21,
                        701.0,
                        336.26
                    ],
                    [
                        690.28,
                        308.05,
                        700.72,
                        336.26
                    ],
                    [
                        690.46,
                        307.89,
                        700.44,
                        336.26
                    ],
                    [
                        690.64,
                        307.72,
                        700.16,
                        336.26
                    ],
                    [
                        690.82,
                        307.56,
                        699.88,
                        336.26
                    ],
                    [
                        691.0,
                        307.4,
                        699.6,
                        336.26
                    ],
                    [
                        691.54,
                        307.4,
                        700.14,
                        336.53
                    ],
                    [
                        692.08,
                        307.4,
                        700.68,
                        336.8
                    ],
                    [
                        692.62,
                        307.4,
                        701.22,
                        337.07
                    ],
                    [
                        693.16,
                        307.4,
                        701.76,
                        337.33
                    ],
                    [
                        693.7,
                        307.4,
                        702.3,
                        337.6
                    ],
                    [
                        694.2,
                        307.4,
                        702.96,
                        338.0
                    ],
                    [
                        694.7,
                        307.4,
                        703.62,
                        338.4
                    ],
                    [
                        695.2,
                        307.4,
                        704.28,
                        338.8
                    ],
                    [
                        695.7,
                        307.4,
                        704.94,
                        339.2
                    ],
                    [
                        696.2,
                        307.4,
                        705.6,
                        339.6
                    ],
                    [
                        697.06,
                        307.48,
                        706.24,
                        339.76
                    ],
                    [
                        697.92,
                        307.56,
                        706.88,
                        339.92
                    ],
                    [
                        698.78,
                        307.64,
                        707.52,
                        340.08
                    ],
                    [
                        699.64,
                        307.72,
                        708.16,
                        340.24
                    ],
                    [
                        700.5,
                        307.8,
                        708.8,
                        340.4
                    ],
                    [
                        701.54,
                        307.16,
                        709.6,
                        340.4
                    ],
                    [
                        702.58,
                        306.52,
                        710.4,
                        340.4
                    ],
                    [
                        703.62,
                        305.88,
                        711.2,
                        340.4
                    ],
                    [
                        704.66,
                        305.24,
                        712.0,
                        340.4
                    ],
                    [
                        705.7,
                        304.6,
                        712.8,
                        340.4
                    ],
                    [
                        706.24,
                        304.6,
                        713.52,
                        340.58
                    ],
                    [
                        706.78,
                        304.6,
                        714.24,
                        340.76
                    ],
                    [
                        707.32,
                        304.6,
                        714.96,
                        340.94
                    ],
                    [
                        707.86,
                        304.6,
                        715.68,
                        341.12
                    ],
                    [
                        708.4,
                        304.6,
                        716.4,
                        341.3
                    ],
                    [
                        709.64,
                        304.6,
                        717.6,
                        341.44
                    ],
                    [
                        710.88,
                        304.6,
                        718.8,
                        341.58
                    ],
                    [
                        712.12,
                        304.6,
                        720.0,
                        341.72
                    ],
                    [
                        713.36,
                        304.6,
                        721.2,
                        341.86
                    ],
                    [
                        714.6,
                        304.6,
                        722.4,
                        342.0
                    ],
                    [
                        715.66,
                        304.6,
                        723.64,
                        342.64
                    ],
                    [
                        716.72,
                        304.6,
                        724.88,
                        343.28
                    ],
                    [
                        717.78,
                        304.6,
                        726.12,
                        343.92
                    ],
                    [
                        718.84,
                        304.6,
                        727.36,
                        344.56
                    ],
                    [
                        719.9,
                        304.6,
                        728.6,
                        345.2
                    ],
                    [
                        721.14,
                        304.6,
                        730.38,
                        345.8
                    ],
                    [
                        722.38,
                        304.6,
                        732.16,
                        346.4
                    ],
                    [
                        723.62,
                        304.6,
                        733.94,
                        347.0
                    ],
                    [
                        724.86,
                        304.6,
                        735.72,
                        347.6
                    ],
                    [
                        726.1,
                        304.6,
                        737.5,
                        348.2
                    ],
                    [
                        727.46,
                        304.38,
                        739.12,
                        348.98
                    ],
                    [
                        728.82,
                        304.16,
                        740.74,
                        349.76
                    ],
                    [
                        730.18,
                        303.94,
                        742.36,
                        350.54
                    ],
                    [
                        731.54,
                        303.72,
                        743.98,
                        351.32
                    ],
                    [
                        732.9,
                        303.5,
                        745.6,
                        352.1
                    ],
                    [
                        734.4,
                        303.4,
                        747.98,
                        352.98
                    ],
                    [
                        735.9,
                        303.3,
                        750.36,
                        353.86
                    ],
                    [
                        737.4,
                        303.2,
                        752.74,
                        354.74
                    ],
                    [
                        738.9,
                        303.1,
                        755.12,
                        355.62
                    ],
                    [
                        740.4,
                        303.0,
                        757.5,
                        356.5
                    ],
                    [
                        742.44,
                        303.0,
                        759.82,
                        357.28
                    ],
                    [
                        744.48,
                        303.0,
                        762.14,
                        358.06
                    ],
                    [
                        746.52,
                        303.0,
                        764.46,
                        358.84
                    ],
                    [
                        748.56,
                        303.0,
                        766.78,
                        359.62
                    ],
                    [
                        750.6,
                        303.0,
                        769.1,
                        360.4
                    ],
                    [
                        753.6,
                        302.64,
                        771.74,
                        361.32
                    ],
                    [
                        756.6,
                        302.28,
                        774.38,
                        362.24
                    ],
                    [
                        759.6,
                        301.92,
                        777.02,
                        363.16
                    ],
                    [
                        762.6,
                        301.56,
                        779.66,
                        364.08
                    ],
                    [
                        765.6,
                        301.2,
                        782.3,
                        365.0
                    ],
                    [
                        768.68,
                        300.08,
                        785.81,
                        365.99
                    ],
                    [
                        771.76,
                        298.96,
                        789.31,
                        366.97
                    ],
                    [
                        774.84,
                        297.84,
                        792.82,
                        367.96
                    ],
                    [
                        777.92,
                        296.72,
                        796.33,
                        368.95
                    ],
                    [
                        781.0,
                        295.6,
                        799.83,
                        369.93
                    ],
                    [
                        784.84,
                        295.12,
                        803.95,
                        371.27
                    ],
                    [
                        788.68,
                        294.64,
                        808.06,
                        372.6
                    ],
                    [
                        792.52,
                        294.16,
                        812.17,
                        373.93
                    ],
                    [
                        796.36,
                        293.68,
                        816.29,
                        375.27
                    ],
                    [
                        800.2,
                        293.2,
                        820.4,
                        376.6
                    ],
                    [
                        803.7,
                        292.28,
                        825.48,
                        378.28
                    ],
                    [
                        807.2,
                        291.36,
                        830.56,
                        379.96
                    ],
                    [
                        810.7,
                        290.44,
                        835.64,
                        381.64
                    ],
                    [
                        814.2,
                        289.52,
                        840.72,
                        383.32
                    ],
                    [
                        817.7,
                        288.6,
                        845.8,
                        385.0
                    ],
                    [
                        821.14,
                        288.06,
                        852.78,
                        387.14
                    ],
                    [
                        824.58,
                        287.52,
                        859.76,
                        389.28
                    ],
                    [
                        828.02,
                        286.98,
                        866.74,
                        391.42
                    ],
                    [
                        831.46,
                        286.44,
                        873.72,
                        393.56
                    ],
                    [
                        834.9,
                        285.9,
                        880.7,
                        395.7
                    ],
                    [
                        842.66,
                        284.26,
                        890.58,
                        398.6
                    ],
                    [
                        850.42,
                        282.62,
                        900.46,
                        401.5
                    ],
                    [
                        858.18,
                        280.98,
                        910.34,
                        404.4
                    ],
                    [
                        865.94,
                        279.34,
                        920.22,
                        407.3
                    ],
                    [
                        873.7,
                        277.7,
                        930.1,
                        410.2
                    ],
                    [
                        885.42,
                        275.18,
                        943.26,
                        414.48
                    ],
                    [
                        897.14,
                        272.66,
                        956.42,
                        418.76
                    ],
                    [
                        908.86,
                        270.14,
                        969.58,
                        423.04
                    ],
                    [
                        920.58,
                        267.62,
                        982.74,
                        427.32
                    ],
                    [
                        932.3,
                        265.1,
                        995.9,
                        431.6
                    ],
                    [
                        950.2,
                        261.68,
                        1012.18,
                        437.84
                    ],
                    [
                        968.1,
                        258.26,
                        1028.46,
                        444.08
                    ],
                    [
                        986.0,
                        254.84,
                        1044.74,
                        450.32
                    ],
                    [
                        1003.9,
                        251.42,
                        1061.02,
                        456.56
                    ],
                    [
                        1021.8,
                        248.0,
                        1077.3,
                        462.8
                    ],
                    [
                        1045.68,
                        243.08,
                        1107.4,
                        474.24
                    ],
                    [
                        1069.56,
                        238.16,
                        1137.5,
                        485.68
                    ],
                    [
                        1093.44,
                        233.24,
                        1167.6,
                        497.12
                    ],
                    [
                        1117.32,
                        228.32,
                        1197.7,
                        508.56
                    ],
                    [
                        1141.2,
                        223.4,
                        1227.8,
                        520.0
                    ],
                    [
                        1173.8,
                        218.3,
                        1270.5,
                        538.1
                    ],
                    [
                        1204.9,
                        211.8,
                        1280.0,
                        559.8
                    ],
                    [
                        1204.9,
                        211.8,
                        1280.0,
                        559.8
                    ]
    ]
    
    bboxes = [
                    [
                        695.16,
                        370.84,
                        708.66,
                        418.42
                    ],
                    [
                        693.58,
                        371.19,
                        707.61,
                        419.03
                    ],
                    [
                        692.0,
                        371.54,
                        706.55,
                        419.65
                    ],
                    [
                        690.42,
                        371.9,
                        705.5,
                        420.27
                    ],
                    [
                        688.84,
                        372.25,
                        704.44,
                        420.89
                    ],
                    [
                        687.27,
                        372.6,
                        703.38,
                        421.51
                    ],
                    [
                        685.69,
                        372.96,
                        702.33,
                        422.13
                    ],
                    [
                        684.11,
                        373.31,
                        701.27,
                        422.74
                    ],
                    [
                        682.53,
                        373.66,
                        700.21,
                        423.36
                    ],
                    [
                        680.95,
                        374.02,
                        699.16,
                        423.98
                    ],
                    [
                        679.37,
                        374.37,
                        698.1,
                        424.6
                    ],
                    [
                        677.77,
                        373.33,
                        696.79,
                        425.22
                    ],
                    [
                        676.17,
                        372.29,
                        695.48,
                        425.84
                    ],
                    [
                        674.57,
                        371.25,
                        694.17,
                        426.46
                    ],
                    [
                        672.97,
                        370.21,
                        692.86,
                        427.08
                    ],
                    [
                        671.37,
                        369.18,
                        691.55,
                        427.7
                    ],
                    [
                        669.77,
                        368.14,
                        690.24,
                        427.04
                    ],
                    [
                        668.17,
                        367.1,
                        688.93,
                        426.38
                    ],
                    [
                        666.57,
                        366.06,
                        687.62,
                        425.72
                    ],
                    [
                        664.97,
                        365.02,
                        686.31,
                        425.06
                    ],
                    [
                        663.37,
                        363.98,
                        685.0,
                        424.4
                    ],
                    [
                        661.45,
                        364.32,
                        684.42,
                        425.11
                    ],
                    [
                        659.52,
                        364.66,
                        683.84,
                        425.82
                    ],
                    [
                        657.6,
                        365.0,
                        683.26,
                        426.53
                    ],
                    [
                        655.7,
                        365.84,
                        682.68,
                        427.24
                    ],
                    [
                        653.8,
                        366.68,
                        682.1,
                        427.95
                    ],
                    [
                        651.9,
                        367.51,
                        681.52,
                        428.66
                    ],
                    [
                        651.45,
                        367.71,
                        680.94,
                        429.37
                    ],
                    [
                        651.0,
                        367.9,
                        680.36,
                        430.08
                    ],
                    [
                        650.56,
                        368.0,
                        679.78,
                        430.79
                    ],
                    [
                        650.11,
                        368.1,
                        679.2,
                        431.5
                    ],
                    [
                        648.87,
                        366.97,
                        675.97,
                        431.77
                    ],
                    [
                        647.64,
                        365.83,
                        672.73,
                        432.04
                    ],
                    [
                        646.4,
                        364.7,
                        669.5,
                        432.31
                    ],
                    [
                        643.85,
                        364.4,
                        667.81,
                        432.58
                    ],
                    [
                        641.3,
                        364.1,
                        666.12,
                        432.85
                    ],
                    [
                        638.76,
                        363.8,
                        664.42,
                        433.12
                    ],
                    [
                        636.21,
                        363.5,
                        662.73,
                        433.39
                    ],
                    [
                        633.66,
                        363.2,
                        661.04,
                        433.66
                    ],
                    [
                        631.11,
                        362.9,
                        659.35,
                        433.93
                    ],
                    [
                        628.57,
                        362.6,
                        657.66,
                        434.2
                    ],
                    [
                        626.72,
                        362.1,
                        657.21,
                        434.67
                    ],
                    [
                        624.88,
                        361.6,
                        656.75,
                        435.14
                    ],
                    [
                        623.03,
                        361.1,
                        656.3,
                        435.61
                    ],
                    [
                        621.19,
                        360.6,
                        654.75,
                        436.08
                    ],
                    [
                        619.34,
                        360.11,
                        653.2,
                        436.55
                    ],
                    [
                        617.5,
                        359.61,
                        649.36,
                        437.02
                    ],
                    [
                        615.65,
                        359.11,
                        645.52,
                        437.49
                    ],
                    [
                        613.81,
                        358.61,
                        641.68,
                        437.96
                    ],
                    [
                        611.96,
                        358.11,
                        637.84,
                        438.43
                    ],
                    [
                        610.12,
                        357.61,
                        634.0,
                        438.9
                    ],
                    [
                        606.23,
                        357.97,
                        632.96,
                        439.83
                    ],
                    [
                        602.35,
                        358.33,
                        631.92,
                        440.76
                    ],
                    [
                        598.47,
                        358.69,
                        630.88,
                        441.69
                    ],
                    [
                        594.58,
                        359.05,
                        629.84,
                        442.62
                    ],
                    [
                        590.7,
                        359.41,
                        628.8,
                        443.55
                    ],
                    [
                        588.56,
                        359.77,
                        627.76,
                        444.48
                    ],
                    [
                        586.42,
                        360.13,
                        626.72,
                        445.41
                    ],
                    [
                        584.28,
                        360.49,
                        625.68,
                        446.34
                    ],
                    [
                        582.14,
                        360.84,
                        624.64,
                        447.27
                    ],
                    [
                        580.0,
                        361.2,
                        623.6,
                        448.2
                    ],
                    [
                        577.56,
                        360.74,
                        619.7,
                        448.59
                    ],
                    [
                        575.12,
                        360.27,
                        615.8,
                        448.98
                    ],
                    [
                        572.68,
                        359.8,
                        609.53,
                        449.37
                    ],
                    [
                        570.24,
                        359.34,
                        603.27,
                        449.76
                    ],
                    [
                        567.8,
                        358.87,
                        597.0,
                        450.15
                    ],
                    [
                        557.0,
                        358.4,
                        595.0,
                        450.54
                    ],
                    [
                        550.9,
                        357.94,
                        594.75,
                        450.93
                    ],
                    [
                        544.8,
                        357.47,
                        594.5,
                        451.32
                    ],
                    [
                        545.55,
                        357.0,
                        594.2,
                        451.71
                    ],
                    [
                        546.3,
                        356.54,
                        594.8,
                        452.1
                    ],
                    [
                        543.88,
                        356.06,
                        592.57,
                        453.51
                    ],
                    [
                        541.45,
                        355.58,
                        590.33,
                        454.92
                    ],
                    [
                        539.03,
                        355.1,
                        588.1,
                        456.33
                    ],
                    [
                        536.61,
                        354.62,
                        585.23,
                        457.74
                    ],
                    [
                        534.18,
                        354.14,
                        582.37,
                        459.15
                    ],
                    [
                        531.76,
                        353.66,
                        579.5,
                        460.56
                    ],
                    [
                        529.34,
                        353.19,
                        573.38,
                        461.97
                    ],
                    [
                        526.91,
                        352.71,
                        567.25,
                        463.38
                    ],
                    [
                        524.49,
                        352.23,
                        561.12,
                        464.79
                    ],
                    [
                        522.07,
                        351.75,
                        555.0,
                        466.2
                    ],
                    [
                        515.54,
                        352.2,
                        552.88,
                        467.79
                    ],
                    [
                        509.01,
                        352.65,
                        550.76,
                        469.38
                    ],
                    [
                        502.48,
                        353.1,
                        548.64,
                        470.97
                    ],
                    [
                        495.96,
                        353.54,
                        546.52,
                        472.56
                    ],
                    [
                        489.43,
                        353.99,
                        544.4,
                        474.15
                    ],
                    [
                        482.9,
                        354.44,
                        542.28,
                        475.74
                    ],
                    [
                        480.28,
                        354.89,
                        540.16,
                        477.33
                    ],
                    [
                        477.65,
                        355.34,
                        538.04,
                        478.92
                    ],
                    [
                        475.03,
                        355.79,
                        535.92,
                        480.51
                    ],
                    [
                        472.4,
                        356.24,
                        533.8,
                        482.1
                    ],
                    [
                        468.13,
                        356.02,
                        531.85,
                        483.36
                    ],
                    [
                        463.86,
                        355.81,
                        529.9,
                        484.62
                    ],
                    [
                        459.59,
                        355.59,
                        521.53,
                        485.88
                    ],
                    [
                        455.32,
                        355.37,
                        513.17,
                        487.14
                    ],
                    [
                        451.05,
                        355.16,
                        504.8,
                        488.4
                    ],
                    [
                        446.78,
                        354.94,
                        501.45,
                        489.66
                    ],
                    [
                        442.51,
                        354.73,
                        498.1,
                        490.92
                    ],
                    [
                        438.24,
                        354.51,
                        494.75,
                        492.18
                    ],
                    [
                        433.97,
                        354.3,
                        491.4,
                        493.44
                    ],
                    [
                        429.7,
                        354.08,
                        489.4,
                        494.7
                    ],
                    [
                        420.27,
                        346.4,
                        487.6,
                        496.61
                    ],
                    [
                        410.83,
                        348.38,
                        483.5,
                        498.52
                    ],
                    [
                        401.4,
                        350.36,
                        479.4,
                        500.43
                    ],
                    [
                        396.87,
                        349.12,
                        471.78,
                        502.34
                    ],
                    [
                        392.34,
                        347.88,
                        464.15,
                        504.25
                    ],
                    [
                        387.81,
                        346.64,
                        456.53,
                        506.16
                    ],
                    [
                        383.29,
                        345.4,
                        448.9,
                        508.07
                    ],
                    [
                        375.59,
                        343.46,
                        439.43,
                        509.98
                    ],
                    [
                        367.9,
                        341.53,
                        429.97,
                        511.89
                    ],
                    [
                        360.2,
                        339.6,
                        420.5,
                        513.8
                    ],
                    [
                        349.88,
                        338.5,
                        413.93,
                        516.41
                    ],
                    [
                        339.57,
                        338.51,
                        407.37,
                        519.02
                    ],
                    [
                        329.25,
                        338.52,
                        400.8,
                        521.63
                    ],
                    [
                        318.93,
                        338.53,
                        394.23,
                        524.24
                    ],
                    [
                        308.62,
                        338.54,
                        387.67,
                        526.85
                    ],
                    [
                        298.3,
                        338.55,
                        381.1,
                        529.46
                    ],
                    [
                        291.95,
                        338.03,
                        377.93,
                        532.07
                    ],
                    [
                        285.6,
                        337.5,
                        374.75,
                        534.68
                    ],
                    [
                        279.25,
                        336.98,
                        371.58,
                        537.29
                    ],
                    [
                        272.9,
                        336.46,
                        368.4,
                        539.9
                    ],
                    [
                        263.28,
                        335.04,
                        359.7,
                        542.92
                    ],
                    [
                        253.67,
                        333.61,
                        351.0,
                        545.94
                    ],
                    [
                        244.05,
                        332.18,
                        337.5,
                        548.96
                    ],
                    [
                        234.44,
                        330.76,
                        324.0,
                        551.98
                    ],
                    [
                        224.82,
                        329.33,
                        310.5,
                        555.0
                    ],
                    [
                        215.2,
                        327.9,
                        297.0,
                        558.02
                    ],
                    [
                        205.59,
                        326.47,
                        283.5,
                        561.04
                    ],
                    [
                        195.97,
                        325.05,
                        276.63,
                        564.06
                    ],
                    [
                        186.36,
                        323.62,
                        269.77,
                        567.08
                    ],
                    [
                        176.74,
                        322.19,
                        262.9,
                        570.1
                    ],
                    [
                        161.02,
                        321.37,
                        256.15,
                        574.42
                    ],
                    [
                        145.3,
                        320.56,
                        249.4,
                        578.74
                    ],
                    [
                        134.62,
                        319.74,
                        239.15,
                        583.06
                    ],
                    [
                        123.95,
                        318.92,
                        228.9,
                        587.38
                    ],
                    [
                        113.27,
                        318.1,
                        218.65,
                        591.7
                    ],
                    [
                        102.6,
                        317.28,
                        208.4,
                        596.02
                    ],
                    [
                        91.92,
                        316.46,
                        192.25,
                        600.34
                    ],
                    [
                        81.25,
                        315.64,
                        176.1,
                        604.66
                    ],
                    [
                        70.57,
                        314.82,
                        159.95,
                        608.98
                    ],
                    [
                        59.9,
                        314.0,
                        143.8,
                        613.3
                    ],
                    [
                        40.17,
                        316.61,
                        124.07,
                        615.91
                    ],
                    [
                        20.44,
                        319.22,
                        104.34,
                        618.51
                    ],
                    [
                        10.22,
                        314.7,
                        91.32,
                        620.0
                    ],
                    [
                        0.0,
                        310.3,
                        78.29,
                        621.49
                    ],
                    [
                        0.0,
                        308.43,
                        68.99,
                        625.94
                    ],
                    [
                        0.0,
                        306.56,
                        59.7,
                        630.4
                    ],
                    [
                        0.0,
                        306.56,
                        36.6,
                        630.4
                    ],
                    [
                        0.0,
                        306.56,
                        36.6,
                        630.4
                    ]
                ]
    
    #find max and min x and y
    for bbox in bboxes:
        if bbox[0] < xMin:
            xMin = bbox[0]
        if bbox[1] < yMin:
            yMin = bbox[1]
        if bbox[2] > xMax:
            xMax = bbox[2]
        if bbox[3] > yMax:
            yMax = bbox[3]
            
    print(xMin, yMin, xMax, yMax)
    image1 = image1[int(yMin):int(yMax), int(xMin):int(xMax)]
    cv2.imwrite("outputs/cropped_image_0013_169.jpg", image1)
#    exit()
    """
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1/255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=True, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)


    frameClone = image1.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 2, colors[i], -1, cv2.LINE_AA)
    #cv2.imshow("Keypoints",frameClone)


    valid_pairs, invalid_pairs = getValidPairs(output)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 1, cv2.LINE_AA)


    cv2.imwrite(output_file,frameClone)
    print("writing frame", starting_frame + counter)

print("done")
    #cv2.imshow("Detected Pose" , frameClone)
    #cv2.waitKey(0)
