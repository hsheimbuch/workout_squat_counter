import os
import json
import time
import math

# Mute OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"]="FATAL"

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms
import torch2trt
import trt_pose.coco
import trt_pose.models

from jetcam.usb_camera import USBCamera

from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

# Define function to calculate euclidean distance between two 2D points
def euclid_dist(p,q):
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))

# Change working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

model_dirname = 'model/'

# Set working image width and height
WIDTH = 224
HEIGHT = 224

# Define display font
font = cv2.FONT_HERSHEY_SIMPLEX

# Show loading screen
loading_screen = np.zeros((HEIGHT*2+140,WIDTH*2,3), np.uint8)
cv2.putText(loading_screen, 'Loading...', 
    (WIDTH-70,HEIGHT+70), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imshow('Workout Squat Counter', loading_screen / 255)
cv2.waitKey(100)

# Load trt_pose human pose task body topology
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

# Check if TensorRT optimized model is already present
optimized_model = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

if not optimized_model in os.listdir(model_dirname):

    # Extract number of body patrs and links between then
    # from human_pose.json
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    # Create trt_pose model
    model = trt_pose.models.resnet18_baseline_att(
        num_parts, 2 * num_links).cuda().eval()

    # Load weights
    model_wights = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

    model.load_state_dict(torch.load(model_dirname + model_wights))

    # Create blank data
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

    # Convert model to TensorRT optimized format using fp16 precision
    model_trt = torch2trt.torch2trt(model, [data], 
        fp16_mode=True, max_workspace_size=1<<25)

    # Save optimized model
    torch.save(model_trt.state_dict(), model_dirname + optimized_model)

# Load TensorRT optimized model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(model_dirname + optimized_model))

# Define image preprocessing function 
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# Define class to parse model return
parse_objects = ParseObjects(topology)
# Define class to visualise model return
draw_objects = DrawObjects(topology)

# Define body keypoint names from COCO-Human-Pose dataset
HIP_LEFT = 11
HIP_RIGHT = 12
KNEE_LEFT = 13
KNEE_RIGHT = 14
FOOT_LEFT = 15
FOOT_RIGHT = 16

# Define function to get individual joints location from peaks
def getJointLocation(peaks, joint_number):
    location_pair = peaks[0][joint_number][0]
    location_y, location_x = location_pair[0], location_pair[1]
    if location_y == 0. and location_x == 0.:
        return None, None    
    return (location_y.item(), location_x.item())

# Define function to determine is the individual joint
# is detected from peaks
def jointVisible(peaks, joint_number):
    if getJointLocation(peaks, joint_number) == (None, None):
        return False
    else:
        return True

# Define function to determine is person is in squatting position
def isSquatting(peaks):
    if jointVisible(peaks, HIP_LEFT):
        if jointVisible(peaks, KNEE_LEFT):
            if getJointLocation(peaks, HIP_LEFT)[0] > \
                getJointLocation(peaks, KNEE_LEFT)[0]:
                return True
            else:
                return False
        if jointVisible(peaks, KNEE_RIGHT):
            if getJointLocation(peaks, HIP_LEFT)[0] > \
                getJointLocation(peaks, KNEE_RIGHT)[0]:
                return True
            else:
                return False
    elif jointVisible(peaks, HIP_RIGHT):
        if jointVisible(peaks, KNEE_LEFT):
            if getJointLocation(peaks, HIP_RIGHT)[0] > \
                getJointLocation(peaks, KNEE_LEFT)[0]:
                return True
            else:
                return False
        if jointVisible(peaks, KNEE_RIGHT):
            if getJointLocation(peaks, HIP_RIGHT)[0] > \
                getJointLocation(peaks, KNEE_RIGHT)[0]:
                return True
            else:
                return False

# Define function to determine is person is in standing position
def isStanding(peaks):
    if jointVisible(peaks, HIP_LEFT):
        if jointVisible(peaks, KNEE_LEFT):
            if getJointLocation(peaks, KNEE_LEFT)[0] - \
                getJointLocation(peaks, HIP_LEFT)[0] > \
                euclid_dist(getJointLocation(peaks, KNEE_LEFT),
                    getJointLocation(peaks, HIP_LEFT)) / 1.25:
                return True
            else:
                return False
        if jointVisible(peaks, KNEE_RIGHT):
            if getJointLocation(peaks, KNEE_RIGHT)[0] - \
                getJointLocation(peaks, HIP_LEFT)[0] > \
                euclid_dist(getJointLocation(peaks, KNEE_RIGHT),
                    getJointLocation(peaks, HIP_LEFT)) / 1.25:
                return True
            else:
                return False
    elif jointVisible(peaks, HIP_RIGHT):
        if jointVisible(peaks, KNEE_LEFT):
            if getJointLocation(peaks, KNEE_LEFT)[0] - \
                getJointLocation(peaks, HIP_RIGHT)[0] > \
                euclid_dist(getJointLocation(peaks, KNEE_LEFT),
                    getJointLocation(peaks, HIP_RIGHT)) / 1.25:
                return True
            else:
                return False
        if jointVisible(peaks, KNEE_RIGHT):
            if getJointLocation(peaks, KNEE_RIGHT)[0] - \
                getJointLocation(peaks, HIP_RIGHT)[0] > \
                euclid_dist(getJointLocation(peaks, KNEE_RIGHT),
                    getJointLocation(peaks, HIP_RIGHT)) / 1.25:
                return True
            else:
                return False


# Define squat couting values
squat_made = False
squat_count = 0

# Set video capture from the USB webcam
camera = USBCamera(capture_device=0)

# Main app loop
while True:
    # Set start timepoint 
    time_start = time.time()
    # Read frame from video capture
    frame = camera.read()
    # Resize frame to match model input size
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # Preprocess frame
    data = preprocess(frame)
    # Process frame by TensorRT model
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    # Parse model output
    counts, objects, peaks = parse_objects(cmap, paf)
    # Visualize model output
    draw_objects(frame, counts, objects, peaks)
    # Resize frame to make display larger
    frame = cv2.resize(frame, (WIDTH*2, HEIGHT*2))
    # Create display image
    display_image = np.zeros((HEIGHT*2+140,WIDTH*2,3), np.uint8)
    display_image[0:HEIGHT*2, 0:WIDTH*2*2] = frame
    if counts == 0:
        cv2.putText(display_image, 'User not in the frame', 
                    (20,HEIGHT*2+40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # Define conditions to assume lower body is visible
    lowerBodyVisible = (jointVisible(peaks, HIP_RIGHT) \
        or jointVisible(peaks, HIP_LEFT)) \
        and (jointVisible(peaks, KNEE_RIGHT) or jointVisible(peaks, KNEE_LEFT))
    if counts > 0 and not lowerBodyVisible: 
        cv2.putText(display_image, 'Make sure lower body', 
                    (20,HEIGHT*2+40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(display_image, 'entirely fits in the frame', 
                    (20,HEIGHT*2+80), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        if isSquatting(peaks):
            squat_made = True
        elif isStanding(peaks):
            if squat_made == True:
                squat_count += 1
                squat_made = False          
    cv2.putText(display_image, 'Num of squats: ' + str(squat_count), 
        (20,HEIGHT*2+120), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    # Set end timepoint 
    time_end = time.time()
    # Calculate fps for current frame
    fps = round(1.0 / (time_end - time_start))
    cv2.putText(display_image, 'FPS: ' + str(fps), 
        (10,HEIGHT*2-10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Display    
    cv2.imshow('Workout Squat Counter', display_image)
    # cv2.waitKey(1)
    k = cv2.waitKey(1)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    
