#!/usr/bin/env python
#-*- coding: utf-8 -*-

import cv2, math, time, sys, os
sys.path.append(os.pardir)
import numpy as np
from my_module.ImageProcess import ImageProcessing
from my_module.HoughDetect import HoughDetect

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from carla_msgs.msg import CarlaEgoVehicleControl

frame = np.empty(shape=[0])

def img_callback(data):
    global frame, bridge

    frame = bridge.imgmsg_to_cv2(data, "bgr8")

def start():
    global frame, bridge

    ip = ImageProcessing()
    hough = HoughDetect()
    bridge = CvBridge()
    
    rospy.init_node('lane_detect')
    rospy.Subscriber('/carla/ego_vehicle/camera/rgb/front/image_color', Image, img_callback, queue_size=1)

    while not rospy.is_shutdown():
        if frame.size == (640*480*3):
            gray = ip.bgr2gray(frame)
            blur = ip.bgr2blur(gray)
            canny = ip.img2canny(blur, 10, 50)
            img = ip.set_roi(canny)
            hough_lines = hough.houghtf(img, 30, 10, 10)
            lines = hough.calculate_lines(img, hough_lines)
            line_img = hough.visualize_lines(frame, lines)
            cv2.imshow('viewer', line_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    try:
        start()
    finally:
        cv2.destroyAllWindows()
