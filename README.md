# Facial Keypoint Detection

This repository contains my first project for the [Computer Vision Nanodegree ](https://www.udacity.com/course/computer-vision-nanodegree--nd891) at [Udacity](https://Udacity.com).  
In this project, I've made a facial keypoint detection system. The system consists of a face detector that uses Haar Cascades and a Convolutional Neural Network (CNN) that predict the facial keypoints in the detected faces. The facial keypoint detection system takes in any image with faces and predicts the location of 68 distinguishing keypoints on each face. I've gone a step further and implemented my model to work with laptop camera.

![example](./output.gif)

## Installing

Model was developed with `Python 3.7.3`. 

```bash
git clone https://github.com/wronaq/FacialKeypointDetection.git
cd FacialKeypointDetection
python -m venv .venv
source .venv/bin/activate
pip install torch==1.6.0 torchvision==0.7.0 opencv-python==4.4.0.40
python detect_from_camera.py False #change to True if you want to save output 
```

LICENSE: This project is licensed under the terms of the MIT license.
