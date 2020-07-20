# Social_Distancing_Analysis_using_OpenCV_and_Yolov3

# Requirements:
1. A video file: to be defined as vid_path in social_distance.py filee.
2. cfg file: contains configuration file of yolo network. Get it from "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg"
3. label names : In data/coco.names which contains names of classes/labels to be detected.  Get it from https://github.com/pjreddie/darknet/blob/master/data/coco.names
4. weights file: I have used yolov3.cfg which returns a network (NET object) to do forward propogation. Get it from https://pjreddie.com/darknet/yolo/

# Key points:
1. findDistance() and isClose() functions performs camera calibration to estimate the parameters of a camera, identifies perspective projection and calibrates using known points required for each video.
2. readNetFromDarknet() reads a network model stored in Darknet model files.
3. getLayerNames() gets the name of all layers of the network.
4. getUnconnectedOutLayers() obtains indexes of the unconnected output layers in order to find out how far function forward() must run through the network.
5. blobFromImage() creates 4-dimensional blob from image i.e. N: batch size,C: channel,H: height,W: width
6. NMSBoxes() performs non maximum suppression given boxes and corresponding scores.

# References:

https://pjreddie.com/darknet/yolo/

https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7

https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

https://medium.com/diaryofawannapreneur/yolo-you-only-look-once-for-object-detection-explained-6f80ea7aaa1e

https://www.learnopencv.com/camera-calibration-using-opencv/

