#!/usr/bin/python3
#-*- coding: utf-8 -*-

from typing import List, Dict

import ctypes
import struct
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


PCD_PATH = 'pcd_02_with_rgb.pcd'

class Open3dHelper(object):
    def load_pcd_data(self, pcd):
        return o3d.io.read_point_cloud(pcd)

    def ros_to_pcd_array(self, pcd):
        pcd = self.load_pcd_data(pcd)
        points = np.zeros((1, 3), dtype=float)
        rgbs = np.zeros((1, 3), dtype=float)

        for p in pcd:
            rgb = p[3]

            s = struct.pack('>f', rgb)
            i = struct.unpack('>l', s)[0]
            pack = ctypes.c_uint32(i).value

            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)

            points = np.append(points, [[p[0], p[1], p[2]]], axis=0)
            rgbs = np.append(rgbs, [[r, g, b]], axis=0)
        return points, rgbs

    def do_draw_pcd(self, pcd: List):
        return o3d.visualization.draw_geometries(pcd, width=640, height=480)

    def do_voxel_downsampling(self, pcd, voxel_size: float=0.05):
        return pcd.voxel_down_sample(voxel_size)

    def do_dbscan_cluster(self, pcd, eps: float=0.02, min_points: int=10, print_progress: bool=True):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress))

        max_label = labels.max()

        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pcd, labels

    def do_backgound_segment(self, pcd, distance_threshold: float=0.1, ransac_n: int=3, num_iterations: int=1000):
        plane_model, inliers = pcd.segment_plane(distance_threshold,
                                             ransac_n,
                                             num_iterations)
        [a, b, c, d] = plane_model
        #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0.0, 0.0, 1.0])
        return inlier_cloud, outlier_cloud

    def do_pcd_to_array(self, pcd):
        return np.asarray(pcd.points), np.asarray(pcd.colors)


    def do_draw_bounding_box(self, labels, points, rgbs):
        obj_points: Dict = {}
        obj_colors: Dict = {}

        for idx, label in enumerate(labels):
            if label >= 0:
                if label not in obj_points:
                    obj_points[label] = np.array([points[idx]])
                    obj_colors[label] = np.array([rgbs[idx]])
                else:
                    obj_points[label] = np.append(obj_points[label], [points[idx]], axis=0)
                    obj_colors[label] = np.append(obj_colors[label], [rgbs[idx]], axis=0)

        draw_list: List = []
        o3dpc: Dict = {}

        for key in obj_points:
            o3dpc[key]: List = []
            '''
            obj[label] = [PCD, bb, bb_center, bb_axis, bb_extent/length]
            '''
            o3dpc[key].append(o3d.geometry.PointCloud())
            o3dpc[key][0].points = o3d.utility.Vector3dVector(obj_points[key])
            o3dpc[key][0].colors = o3d.utility.Vector3dVector(obj_colors[key])
            o3dpc[key].append(o3dpc[key][0].get_axis_aligned_bounding_box())
            o3dpc[key].append(o3dpc[key][0].get_center())
            o3dpc[key].append(np.array(o3dpc[key][1].get_box_points()))
            o3dpc[key].append(o3dpc[key][1].get_extent())
            o3dpc[key][1].color = (0, 1, 0)
            draw_list.append(o3dpc[key][0])
            draw_list.append(o3dpc[key][1])

        #self.do_draw_pcd(draw_list)

        return o3dpc, draw_list

if __name__ == '__main__':
    o3dh = Open3dHelper()
    pcd = o3dh.load_pcd_data(PCD_PATH)
    pcd = o3dh.do_voxel_downsampling(pcd, 0.1)
    print(f'DonwSampling: {pcd}')
    bg, obj = o3dh.do_backgound_segment(pcd, 0.2, 3, 1000)
    #print(f'BackGround: {bg}, \nObject: {obj}')
    obj_cluster, labels = o3dh.do_dbscan_cluster(obj, 1.0, 10, False)
    #o3dh.do_draw_pcd([obj_cluster])
    point_array, rgb_array = o3dh.do_pcd_to_array(obj_cluster)
    o3dh.do_draw_bounding_box(labels, point_array, rgb_array)
