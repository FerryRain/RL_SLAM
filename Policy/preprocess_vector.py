#!/usr/bin/env python3
"""
# File       : preprocess_vector.py
# Time       ：12/22/22 8:23 AM
# Author     ：Kust Kenny
# version    ：python 3.6 
# Description：
"""
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from hwt_data.msg import Hwt_ht_basic
import torch

global color_image, depth_image, basic

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
    rospy.init_node('get_image', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    # rospy.Subscriber("/camera/depth/image_raw", Image, callback2)
    rospy.Subscriber("/hwt/px4/basic", Hwt_ht_basic, callback3)
    r = rospy.Rate(10)  # 10Hz

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
