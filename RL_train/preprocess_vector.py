#!/usr/bin/env python3
"""
# File       : preprocess_vector.py
# Time       ：12/22/22 8:23 AM
# Author     ：Kust Kenny
# version    ：python 3.6 
# Description：
"""
import argparse
from pathlib import Path

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from hwt_data.msg import Hwt_ht_basic
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from Environment_preprocessing.test_yolo import mainloop, args_set, plot_frame

from Thirdparty.yolov7.models.experimental import attempt_load
from Thirdparty.yolov7.utils.datasets import LoadStreams, LoadImages
from Thirdparty.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from Thirdparty.yolov7.utils.plots import plot_one_box
from Thirdparty.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from Thirdparty.yolov7.utils.datasets import letterbox

import sys
sys.path.append('../Thirdparty/yolov7')

global color_image, depth_image, basic
global old_img_b, old_img_h, old_img_w

def args_set():
    """
    Initialization Parameter
    Returns:
        Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default="2", help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='true', help='display results')
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


# detect model init and get
def plot_frame(pred, img1, img, names, colors):
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

        cv2.imshow("Vision", img)
        cv2.waitKey(1)  # 1 millisecond

def detect(img, model, imgsz, device, half, view_img, names, colors, stride):
    global old_img_b, old_img_h, old_img_w
    img1 = letterbox(img, imgsz, stride=stride)[0]
    img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img1 = np.ascontiguousarray(img1)

    img1 = torch.from_numpy(img1).to(device)
    img1 = img1.half() if half else img1.float()  # uint8 to fp16/32
    img1 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img1.ndimension() == 3:
        img1 = img1.unsqueeze(0)

    if device.type != 'cpu' and (
            old_img_b != img1.shape[0] or old_img_h != img1.shape[2] or old_img_w != img1.shape[3]):
        old_img_b = img1.shape[0]
        old_img_h = img1.shape[2]
        old_img_w = img1.shape[3]
        for i in range(3):
            model(img1, augment=args.augment)[0]

    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img1, augment=args.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes,
                               agnostic=args.agnostic_nms)
    t3 = time_synchronized()

    # print(pred)

    if view_img:
        plot_frame(pred, img1, img, names, colors)

    return pred

def init_model():
    global old_img_b, old_img_h, old_img_w
    # Init Yolo Model
    source, weights, view_img, save_txt, imgsz, trace = args.source, args.weights, args.view_img, \
        args.save_txt, args.img_size, not args.no_trace
    set_logging()
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    device = select_device(args.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, args.img_size)

    if half:
        model.half()  # to FP16

    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1
    return model, imgsz, device, half, view_img, webcam, names, colors, stride

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

    # Init Model
    model, imgsz, device, half, view_img, webcam, names, colors, stride = init_model()

    while True:
        try:
            # cv2.imshow('color_image', color_image)
            # cv2.imshow('depth_image', depth_image)
            # print(basic.attitude[2])
            pred = detect(color_image, model, imgsz, device, half, view_img, names, colors, stride)
            print(pred[0][0])
        except:
            pass
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            rospy.signal_shutdown("shut_down")
            break
            
    rospy.spin()
