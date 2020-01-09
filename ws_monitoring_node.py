import os
import cv2
import time
import argparse
import torch
import numpy as np
from distutils.util import strtobool

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

import rospy
from std_msgs.msg import Int32
from ws_monitoring_msgs.msg import (location_2d, locations_2d)

pub = rospy.Publisher('object_location', locations_2d, queue_size=1)

class Tracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            print ("Running on CPU!")
        else:
            print ("Running on CUDA!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        rospy.init_node('workspace_monitoring_node')

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Original resolution: " + str(self.im_width) + "x" + str(self.im_height))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        while self.vdo.grab(): 
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            ori_im = cv2.resize(ori_im, (640, 480))
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

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
                    self.publishCallback(array)
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                    ori_im = self.drawPoints(array, ori_im)

            end = time.time()
            #print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
        return idx_frame

    def publishCallback(self, array):
        people = locations_2d()
        people.header.frame_id = "/camera_link"
        people.header.stamp = rospy.Time.now()
        people.objects = array
        pub.publish(people)

    def drawPoints(self, array, image):
        for i in range (0, len(array)):
            center_coordinates = (int(array[i].i), int(array[i].j))
            color = (0, 0, 255)
            image = cv2.circle(image, center_coordinates, 4, color, -1)
        return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str, default="../workspace_monitoring_dl/input_vid/demo3.avi")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with Tracker(cfg, args) as trk:
        t_start = time.time()
        n_frames = trk.run()
        t_end = time.time()
        duration = t_end - t_start
        avg_fps_rate = n_frames / duration
        print("Average FPS rate: {:.03f}".format(avg_fps_rate))
    rospy.is_shutdown()
