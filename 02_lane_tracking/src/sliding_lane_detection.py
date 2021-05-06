#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import cv2, math, time, sys, os
sys.path.append(os.pardir)
import numpy as np
from my_module.ImageProcess import ImageProcessing
from my_module.SlideDetect import SlideDetect

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
    sliding = SlideDetect()
    bridge = CvBridge()
    
    rospy.init_node('lane_detect')
    rospy.Subscriber('/carla/ego_vehicle/rgb_front/image', Image, img_callback, queue_size=1)

    while not rospy.is_shutdown():
        if frame.size == (640*480*3):
            blur = ip.bgr2blur(frame)
            l_color = ip.img2hls(blur, 220, 255)
            warp_img, M, Minv = ip.perspective_img(l_color)
            left, right = sliding.histogram(warp_img)
            warp_img = sliding.SlidingWindow(warp_img, left, right, 10)
            cv2.imshow('viewer', warp_img)
            out_img = sliding.draw_lane(frame, warp_img, Minv)
            cv2.imshow('result', out_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    try:
        start()
    finally:
        cv2.destroyAllWindows()
