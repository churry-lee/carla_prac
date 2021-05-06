#!/usr/bin/env python

import rospy, carla, random
import numpy as np
import pickle

green = carla.Color(0, 255, 0)
cyan = carla.Color(0, 255, 255)

trail_life_time = 1000
waypoint_separation = 2
map_len = 1000
# setting starting point
sx, sy, sz = 35.5, 203.8, 1.0

def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=1):

    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False)

def making_traj(m, debug):

    loc = carla.Location(sx, sy, sz)
    current_w = m.get_waypoint(loc)

    traj_x = [current_w.transform.location.x]
    traj_y = [current_w.transform.location.y]
    traj_yaw = [np.radians(current_w.transform.rotation.yaw)]

    while len(traj_x) <= map_len:

        # list of potential next waypoints
        potential_w = list(current_w.next(waypoint_separation))
        next_w = random.choice(potential_w)
        wp_x = next_w.transform.location.x
        wp_y = next_w.transform.location.y
        wp_yaw = np.radians(next_w.transform.rotation.yaw)

        traj_x.append(wp_x)
        traj_y.append(wp_y)
        traj_yaw.append(wp_yaw)

        draw_waypoint_union(debug, current_w, next_w, cyan if current_w.is_junction else green, trail_life_time)

        potential_w.remove(next_w)
        current_w = next_w

    traj = [traj_x, traj_y, traj_yaw]

    #with open('traj.pkl', 'wb') as f:
    #    pickle.dump(traj, f)

    return traj


def main():
    global client, world, m, debug

    client = carla.Client('localhost', 2000)
    world = client.get_world()
    m = world.get_map()
    debug = world.debug

    making_traj(m, debug)


if __name__ == "__main__":
    main()
