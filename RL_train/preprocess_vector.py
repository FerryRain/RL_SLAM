#!/usr/bin/env python3
"""
# File       : preprocess_vector.py
# Time       ：12/22/22 8:23 AM
# Author     ：Kust Kenny
# version    ：python 3.6 
# Description：
"""
import argparse
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from hwt_data.msg import Hwt_ht_basic
import torch
from Environment_preprocessing.test_yolo import mainloop, args_set, plot_frame

import sys
sys.path.append('../Thirdparty/yolov7')

global color_image, depth_image, basic

def args_set():
    """
    Initialization Parameter
    Returns:
        Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/kenny/RL_SLAM/Environment_preprocessing/yolov7-tiny.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default="2", help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    return parser.parse_args()

def callback(data1):
    global color_image
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(data1, 'bgr8')

    # cv2.imshow('color_image', color_image)
    # cv2.waitKey(1)

# def callback2(data2):
#     global depth_image
#     bridge = CvBridge()
#     depth_image = bridge.imgmsg_to_cv2(data2, '16UC1')
#     # cv2.imshow('depth_image', depth_image)
#     # cv2.waitKey(1)

def callback3(data3):
    global basic
    basic = data3



if __name__ == '__main__':
    global color_image, depth_image, basic
    args = args_set()

    # Init ROS
    rospy.init_node('get_image', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    # rospy.Subscriber("/camera/depth/image_raw", Image, callback2)
    rospy.Subscriber("/hwt/px4/basic", Hwt_ht_basic, callback3)
    r = rospy.Rate(10)  # 10Hz


    mainloop(args)
    # x,y,z,vx,vy,yaw(attitude[2])
    while True:
        try:
            cv2.imshow('color_image', color_image)
            # cv2.imshow('depth_image', depth_image)
            print(basic.attitude[2])
        except:
            pass
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            rospy.signal_shutdown("shut_down")
            break

        # print("1")
        # r.sleep()
    rospy.spin()
