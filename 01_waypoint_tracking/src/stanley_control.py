#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy, math, carla
import numpy as np

from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
# make waypoint
from make_traj import making_traj

# setting starting point
sx, sy, sz, syaw = 20.5, 210.8, 1.0, 0.0
# intialize current ego_vehicle state
curr_pos = {"DX":0.0, "DY":0.0, "DZ":0.0, "AX":0.0, "AY":0.0, "AZ":0.0, "AW":0.0}
curr_roll, curr_pitch, curr_yaw = 0.0, 0.0, 0.0
curr_vel = 0.0

target_speed = 20.0 / 3.6      # [m/s]
k = 1.0                        # control gain
L = 2.875                      # vehicle's wheelbase
max_steer = np.radians(30.0)
Kp = 0.5                       # speed P-gain

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def stanley_control(x, y, yaw, v, map_xs, map_ys, map_yaws):
    # find nearest waypoint
    min_dist = 1e9
    min_index = 0
    n_points = len(map_xs)

    # convert rear_wheel x-y coordinate to front_wheel x-y coordinate
    # L = wheel base
    front_x = x + L * np.cos(yaw)
    front_y = y + L * np.sin(yaw)

    for i in range(n_points):
        # calculating distance (map_xs, map_ys) - (front_x, front_y)
        dx = front_x - map_xs[i]
        dy = front_y - map_ys[i]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < min_dist:
            min_dist = dist
            min_index = i

    # compute cte at front axle
    print("min_index: {}".format(min_index))
    map_x = map_xs[min_index]
    map_y = map_ys[min_index]
    map_yaw = map_yaws[min_index]
    # nearest x-y coordinate (map_x, map_y) - front_wheel coordinate (front_x, front_y)
    dx = map_x - front_x
    dy = map_y - front_y

    perp_vec = [np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)]
    cte = np.dot([dx, dy], perp_vec)

    # control law
    # heading error = yaw_term
    yaw_term = normalize_angle(map_yaw - yaw)
    cte_term = np.arctan2(k*cte, v)

    # steering
    steer = yaw_term + cte_term
    print("yaw_term: {}, cte_term: {}".format(yaw_term, cte_term))
    print("wp_x: {}, wp_y: {}".format(map_x, map_y, map_yaw))

    return steer


def status_callback(data):
    global curr_vel

    curr_vel = data.velocity


def pos_callback(data):
    global curr_pos, curr_yaw

    curr_pos["DX"] = data.pose.pose.position.x
    curr_pos["DY"] = data.pose.pose.position.y
    curr_pos["DZ"] = data.pose.pose.position.z
    curr_pos["AX"] = data.pose.pose.orientation.x
    curr_pos["AY"] = data.pose.pose.orientation.y
    curr_pos["AZ"] = data.pose.pose.orientation.z
    curr_pos["AW"] = data.pose.pose.orientation.w

    (curr_roll, curr_pitch, curr_yaw) = euler_from_quaternion((curr_pos["AX"], curr_pos["AY"], curr_pos["AZ"], curr_pos["AW"]))


def main():
    global curr_vel

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    m = world.get_map()
    debug = world.debug

    ego_control = CarlaEgoVehicleControl()
    ego_status = CarlaEgoVehicleStatus()

    rospy.init_node('driver', anonymous=True)
    pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)
    rospy.Subscriber('/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, status_callback, queue_size = 1)
    rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, pos_callback, queue_size = 1)

    waypoints = making_traj(m, debug)
    wp_xs = waypoints[0]
    wp_ys = waypoints[1]
    wp_yaws = waypoints[2]

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # generate acceleration, and steering
        speed_error = target_speed - curr_vel
        acc = Kp * speed_error
        # caution y and yaw sign
        steer = stanley_control(curr_pos["DX"], -curr_pos["DY"], -curr_yaw, curr_vel, wp_xs, wp_ys, wp_yaws)
        print("x: {}, y: {}, yaw: {}".format(curr_pos["DX"], -curr_pos["DY"], -np.degrees(curr_yaw)))
        # Publishing Topic
        ego_control.throttle = acc
        ego_control.steer = steer
        pub.publish(ego_control)
        print("steering: {:.3f}".format(steer))
        print('\n')
        rate.sleep()

if __name__ == "__main__":
    main()

