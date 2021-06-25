#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import ctypes
import struct
import open3d as o3d
import numpy as np

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import tf
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion

from open3d_tutorial import Open3dHelper

def pose_to_marker_msg(obj_dic):
    for key, val in obj_dic.items():
        len_x, len_y, len_z = val[4][0], val[4][1], val[4][2]
        #print(f'Length: {len_x}, {len_y}, {len_z}')
        cent_x, cent_y, cent_z = val[2][0], val[2][1], val[2][2]
        #print(f'Center: {cent_x}, {cent_y}, {cent_z}')


        #quat = tf.transformations.quaternion_from_euler(0, 0, np.arctan2(val[4][1], val[4][0]))
        quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        m = Marker()
        m.header.frame_id = "ego_vehicle"
        m.header.stamp = rospy.Time.now()
        m.id = key
        m.type = 1

        m.pose.position.x = cent_x
        m.pose.position.y = cent_y
        m.pose.position.z = cent_z + 2.4
        m.pose.orientation = Quaternion(*quat)
        #m.pose.orientation = 0

        m.scale.x = len_x
        m.scale.y = len_y
        m.scale.z = len_z

        m.color.r = 255.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.9

    return m

def main(pcd):
    pcd = o3dh.do_voxel_downsampling(pcd, 0.05)
    bg, obj = o3dh.do_backgound_segment(pcd, 0.1, 3, 1000)
    #o3dh.do_draw_pcd([bg, obj])
    obj_cluster, labels = o3dh.do_dbscan_cluster(obj, 1.8, 10, False)
    #o3dh.do_draw_pcd([obj_cluster])
    point_array, rgb_array = o3dh.do_pcd_to_array(obj_cluster)
    obj_dic, draw_list = o3dh.do_draw_bounding_box(labels, point_array, rgb_array)
    #o3dh.do_draw_pcd(draw_list)
    #o3dh.do_draw_pcd([draw_list[2], draw_list[3]])
    #print(obj_dic[2])
    marker = pose_to_marker_msg(obj_dic)
    bb_pub.publish(marker)
    obj_dic = {}


def pcl_callback(msg):
    clouds = pc2.read_points(msg, skip_nans=True)
    #clouds = o3d.io.read_point_cloud(msg)
    points = np.zeros((1, 3), dtype=np.float)
    rgbs = np.zeros((1, 3), dtype=np.float)

    for cloud in clouds:
        rgb = cloud[3]

        s = struct.pack('>f', rgb)
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        points = np.append(points, [[cloud[0], cloud[1], cloud[2]]], axis=0)
        rgbs = np.append(rgbs, [[r, g, b]], axis=0)

    o3dpc = o3d.geometry.PointCloud()
    o3dpc.points = o3d.utility.Vector3dVector(points[1:])
    o3dpc.colors = o3d.utility.Vector3dVector(rgbs[1:])

    main(o3dpc)

if __name__ == '__main__':
    o3dh = Open3dHelper()

    rospy.init_node('Open3d_PCL')
    rospy.Subscriber('/carla/ego_vehicle/lidar', PointCloud2, pcl_callback, queue_size=1)
    bb_pub = rospy.Publisher('object_marker', Marker, queue_size=1)

    rospy.spin()
