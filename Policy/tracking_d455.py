#!/usr/bin/env python3
# coding:utf-8
"""
# File       : tracking_d455.py
# Time       ：10/21/22 5:25 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np
import argparse
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge




OPENCV_OBJECT_TRACKERS = {
    "kcf":cv2.TrackerKCF_create,
    "boost": cv2.legacy.TrackerBoosting_create,
    "csrt": cv2.TrackerCSRT_create,
    "goturn":cv2.TrackerGOTURN_create,
    "mil":cv2.TrackerMIL_create(),
    "mosse":cv2.legacy.TrackerMOSSE_create
}

def init_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t","--tracker",type=str,default="kcf",help="please choose the tracker algorithm")
    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = init_arg()
    trackers = cv2.legacy.MultiTracker_create()
    rospy.init_node("listener", anonymous=True)
    # point = Point()
    # image_point_pubulish = rospy.Publisher('/camera/point', Point, queue_size=1)
    pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    pub2 = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=1)
    bridge = CvBridge()
    _ = True

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    while _:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        (h,w) = frame.shape[:2]
        width = 600
        r = width/float(w)
        frame = cv2.resize(frame,(width,int(h*r)))

        (success,boxes) = trackers.update(frame)

        # 绘制区域
        try:
            for box in boxes:
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                a = 0
                i = 0
                for x1 in range(int(x+w-10),int(x+w+10)):
                    for y1 in range(int(y+w-10),int(y+w+10)):
                        a += depth_image[int(y1), int(x1)]
                        # a += depth_image[int(y1 + h / 2), int(x1 + w/2)]
                        #print(depth_image[int(y + h / 2), int(x + w/2)])
                        i += 1
                distance =  a/i
                print(a / i)
                # point.x = x + w/2
                # point.y = y + h/2
                # point.z = distance
                # image_point_pubulish.publish(point)
        except:
            pass




        cv2.imshow("press to select a box",frame)
        cv2.imshow("depth", depth_colormap)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            # 选择一个区域
            box = cv2.selectROI("press to select a box",frame,fromCenter=False,showCrosshair=True)

            # 创建一个新的追踪器
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            trackers.add(cv2.legacy.TrackerKCF_create(),frame,box)
        elif key == 27:
            break
        pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        pub2.publish(bridge.cv2_to_imgmsg(depth_image, "16UC1"))
    cv2.destroyAllWindows()
