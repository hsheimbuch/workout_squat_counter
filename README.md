# Description
Count and display number of squat excercises during workout using Jetson Nano
and compatible USB Webcam.

# Background
This app is designed to help count squat exercises using Jetson Nano
and compatible USB Webcam. It uses trt_pose to detect body keypoints
and then processes their relative position.

The keypoints are visualised and displayed, along with the excercise count
and warning messages (in case the body is not properly positioned
in the frame).

The app operates at about 12 FPS of Jetson Nano.

# Hardware requirements
Jetson Nano and Jetson Nano compatible USB camera.

# Software requirements
1. Installed [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack)
2. Installed [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) with all prerequisites
3. Installed [Jetcam](https://github.com/NVIDIA-AI-IOT/jetcam)

## Install

```
$ git clone git@github.com:hsheimbuch/workout_squat_counter.git
```

Download the [weights](https://drive.google.com/file/d/1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd/view) and place them in 'model/'.

## Run program
The camera should be placed on the floor at ~2m distance
from the the person. Only one person must be in frame! 

Loading takes about 1 minute.

Pressing ESC key closes the program.

```
$ python3 workout_squat_counter.py
``` 

# Demonstration
[Video demonstration](https://youtu.be/Q3hCjoGjmb0)

# Maintainer
Heinrich Heimbuch @hsheimbuch

# Contributing
Heinrich Heimbuch @hsheimbuch

# Licence 
Free for non-commercial usage
