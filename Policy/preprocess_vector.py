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


def callback(data1,data2):
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(data1, 'bgr8')
    depth_image = bridge.imgmsg_to_cv2(data2, '16UC1')
    cv2.imshow('color_image',color_image)
    cv2.waitKey(10)



if __name__ == '__main__':
    rospy.init_node('get_image', anonymous=True)
    color = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth = message_filters.Subscriber("/camera/depth/image_raw", Image)
    color_depth = message_filters.TimeSynchronizer([color, depth], 1)  # 绝对时间同步
    color_depth.registerCallback(callback)
    rospy.spin()