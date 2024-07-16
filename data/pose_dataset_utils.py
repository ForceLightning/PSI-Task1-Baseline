def get_labels(file_path):
    labels = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            label = line.split()[0]
            labels.append(int(label))

    return labels

def get_boxes(file_path):
    bboxes = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            xcenter = line.split()[1]
            ycenter = line.split()[2]
            xwidth = line.split()[3]
            ywidth = line.split()[4]
            xwidth_half = float(xwidth) / 2
            xstart = float(xcenter) - float(xwidth_half)
            xend = float(xcenter) + float(xwidth_half)
            ywidth_half = float(ywidth) / 2
            ystart = float(ycenter) - float(ywidth_half)
            yend = float(ycenter) + float(ywidth_half)
            bboxes.append([xstart, ystart, xend, yend])
    return bboxes

def get_keypoints(file_path, start= 5):
    keypoints = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines: 
            kp_list = list(map(float, line.split()))
            keypoints.append(kp_list[start:])
    return keypoints
    