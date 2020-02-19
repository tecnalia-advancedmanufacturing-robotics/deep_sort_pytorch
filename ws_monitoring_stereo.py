#!/home/tecnalia/Workspace/human_detection_ws/src/venv/bin/python
import os
import sys, time
import numpy as np
from scipy.ndimage import filters
import cv2
import argparse
import torch
from distutils.util import strtobool

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from ws_monitoring_msgs.msg import (location_2d, locations_2d)

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

idx_frame = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("LEFT_IMAGE_TOPIC", type=str, default="/stereo/left/image_raw/compressed")
    parser.add_argument("RIGHT_IMAGE_TOPIC", type=str, default="/stereo/right/image_raw/compressed")
    parser.add_argument("LEFT_OUTPUT_TOPIC", type=str, default="/left/locations")
    parser.add_argument("RIGHT_OUTPUT_TOPIC", type=str, default="/right/locations")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()

def show_image(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(1)
    # msg = CompressedImage()
    # msg.header.stamp = rospy.Time.now()
    # msg.format = name
    # msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
    # if name == "left":
    #     left_img_pub.publish(msg)
    # else:
    #     right_img_pub.publish(msg)

def publishCallback(array, camera):
    people = locations_2d()
    people.header.frame_id = "/camera_link"
    people.header.stamp = rospy.Time.now()
    people.objects = array
    if camera == "left":
        pub_left.publish(people)
    else:
        pub_right.publish(people)

def drawPoints(array, image):
    for i in range (0, len(array)):
        center_coordinates = (int(array[i].i), int(array[i].j))
        color = (0, 0, 255)
        image = cv2.circle(image, center_coordinates, 4, color, -1)
    return image

def callback(left_msg, right_msg):
    global idx_frame
    idx_frame += 1

    np_arr_l = np.fromstring(left_msg.data, np.uint8)
    np_arr_r = np.fromstring(right_msg.data, np.uint8)

    ori_left = cv2.imdecode(np_arr_l, cv2.IMREAD_COLOR) # OpenCV >= 3.0
    ori_left = cv2.resize(ori_left, (320, 240))

    ori_right = cv2.imdecode(np_arr_r, cv2.IMREAD_COLOR) # OpenCV >= 3.0
    ori_right = cv2.resize(ori_right, (320, 240))

    im_left = cv2.cvtColor(ori_left, cv2.COLOR_BGR2RGB)
    im_right = cv2.cvtColor(ori_right, cv2.COLOR_BGR2RGB)

    th = idx_frame % args.frame_interval
    if th == 0:
        do_detection(im_left, ori_left, "left", "Left View")
        do_detection(im_right, ori_right, "right", "Right View")

def do_detection(im, ori_im, camera, window):
    bbox_xywh, cls_conf, cls_ids = detector(im)
    if bbox_xywh is not None:
        # select person class
        mask = cls_ids==0

        bbox_xywh = bbox_xywh[mask]
        bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
        cls_conf = cls_conf[mask]

        # do tracking
        outputs = deepsort.update(bbox_xywh, cls_conf, im)

        # draw boxes for visualization
        if len(outputs) > 0:
            bbox_xyxy = outputs[:,:4]
            identities = outputs[:,-1]
            array = []

            for i,box in enumerate(bbox_xyxy):
                x1,y1,x2,y2 = [int(i) for i in box]
                id = identities[i]
                object_x = location_2d()
                object_x.object_id = int(id)
                object_x.i = x1 + (x2-x1)/2
                object_x.j = y2 + (y1-y2)/2
                array.append(object_x)
            publishCallback(array, camera)
            # ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
            # ori_im = drawPoints(array, ori_im)

            # if args.display:
                # ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                # ori_im = drawPoints(array, ori_im)
                # show_image(ori_im, window)
                # print(str(len(array)) + " people were seen in the scene at " + camera + " image.")

# Initiaze DL stuff

# Parsing
args = parse_args()
cfg = get_config()
cfg.merge_from_file(args.config_detection)
cfg.merge_from_file(args.config_deepsort)

# Setup
use_cuda = args.use_cuda and torch.cuda.is_available()
if not use_cuda:
    print ("Running on CPU!")
else:
    print ("Running on CUDA!")

detector = build_detector(cfg, use_cuda=use_cuda)
deepsort = build_tracker(cfg, use_cuda=use_cuda)
class_names = detector.class_names

# Initialize ROS stuff
rospy.init_node('workspace_monitoring_node', anonymous=True)

# Data Publishers
pub_left = rospy.Publisher(args.LEFT_OUTPUT_TOPIC, locations_2d, queue_size=1)
pub_right = rospy.Publisher(args.RIGHT_OUTPUT_TOPIC, locations_2d, queue_size=1)

# Image Publisher
# left_img_pub = rospy.Publisher("/left/image_raw/compressed", CompressedImage, queue_size=1)
# right_img_pub = rospy.Publisher("/right/image_raw/compressed", CompressedImage, queue_size=1)

# Subscribers
sub_right_image = message_filters.Subscriber(args.LEFT_IMAGE_TOPIC, CompressedImage)
sub_left_image = message_filters.Subscriber(args.RIGHT_IMAGE_TOPIC, CompressedImage)
ts = message_filters.ApproximateTimeSynchronizer([sub_left_image, sub_right_image], 1, 0.1)
ts.registerCallback(callback)

print("ROS network initialized!")
t_start = time.time()

# Loop
while not rospy.is_shutdown():
    rospy.spin()

t_end = time.time()

# Summary
duration = t_end - t_start
avg_fps_rate = idx_frame / duration
print("\n")
print("Runtime: {:.03f}".format(duration))
print("Frames transmitted: " + str(idx_frame))
print("Average FPS rate: {:.03f}".format(avg_fps_rate))