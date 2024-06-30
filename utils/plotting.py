import numpy as np
import cv2
import math
import torch

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class PosePlotter:
    def __init__(self):

        self.pose_palette = np.array(
                    [
                        [255, 128, 0],
                        [255, 153, 51],
                        [255, 178, 102],
                        [230, 230, 0],
                        [255, 153, 255],
                        [153, 204, 255],
                        [255, 102, 255],
                        [255, 51, 255],
                        [102, 178, 255],
                        [51, 153, 255],
                        [255, 153, 153],
                        [255, 102, 102],
                        [255, 51, 51],
                        [153, 255, 153],
                        [102, 255, 102],
                        [51, 255, 51],
                        [0, 255, 0],
                        [0, 0, 255],
                        [255, 0, 0],
                        [255, 255, 255],
                    ],
                    dtype=np.uint8,
                )
        self.skeleton = [
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                ]
        self.limb_color = self.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = self.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def plot_keypoints(self, image, keypoints, shape=(640, 640), radius=5):
        for kpt in keypoints: 
            for point in kpt:
                for i, p in enumerate(point):
                    color_k = [int(x) for x in self.kpt_color[i]]
                    x_coord = int(p[0])
                    y_coord = int(p[1])
                    conf = p[2]
                    if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                        if conf < 0.5:
                                continue
                        cv2.circle(image, (x_coord, y_coord), radius=radius, color=color_k, 
                                    thickness=-1, lineType=cv2.LINE_AA)

        for kpt in keypoints: 
            for point in kpt:
                ndim = point.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    pos1 = (int(point[(sk[0] - 1), 0]), int(point[(sk[0] - 1), 1]))
                    pos2 = (int(point[(sk[1] - 1), 0]), int(point[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = point[(sk[0] - 1), 2]
                        conf2 = point[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(image, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, 
                                lineType=cv2.LINE_AA)
                    
    def plot_masks(self, image, masks):
        for mask in masks:
            contours = np.array(mask, dtype=np.int32)
            contours = contours.reshape((-1, 1, 2))
            cv2.drawContours(image, [contours], -1, (0, 255, 0), cv2.FILLED)
                          
def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

def region_of_interest(img, vertices): 
    mask = np.zeros_like(img)        
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask) 
    return masked_image

def draw_the_lines(img, lines): 
    if lines is None:
        return
    
    imge = np.copy(img)     
    blank_image = np.zeros((imge.shape[0], imge.shape[1], 3),
                            dtype=np.uint8)
    
    for line in lines:  
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            imge = cv2.addWeighted(imge, 0.8, blank_image, 1, 0.0) 
    return imge
    
def road_lane_detection(orig_image):
    # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = orig_image
    height= image.shape[0]
    width= image.shape[1]
    region_of_interest_coor = [(0, 600), 
                               (width / 2, height / 2),
                               (width, 600)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)

    cropped = region_of_interest(canny_image,
                                 np.array([region_of_interest_coor], np.int32))

    lines = cv2.HoughLinesP(cropped, rho=6, theta=np.pi/60, threshold=160,
                            lines=np.array([]), minLineLength=40, maxLineGap=25)
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    print(lines)
    for line in lines:
        for x1, y1, x2, y2 in line:     
            slope = (y2 - y1) / (x2 - x1) 
        if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
            continue
        if slope <= 0: # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else: # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5)) # <-- Just below the horizon
    max_y = int(image.shape[0]) # <-- The bottom of the image  
    if left_line_x == [] or left_line_y == [] or right_line_x == [] or right_line_y == []:
        return orig_image
    
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    
    image_with_lines = draw_the_lines(
        image, 
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]]
        ) 
    
    return image_with_lines

def overlay(image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.
        
        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        # color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined