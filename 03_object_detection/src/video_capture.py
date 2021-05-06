import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from carla_msgs.msg import CarlaEgoVehicleControl

frame = np.empty(shape=[0])
video_name = 'test5.avi'

def img_callback(data):
    global frame, bridge

    frame = bridge.imgmsg_to_cv2(data, "bgr8")

def start():
    global frame, bridge

    bridge = CvBridge()

    rospy.init_node('capture')
    rospy.Subscriber('/carla/ego_vehicle/camera/rgb/front/image_color', Image, img_callback, queue_size=1)

    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 480))
    
    while not rospy.is_shutdown():
        if frame.size == (640*480*3):
            cv2.imshow('show', frame)
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    try:
        start()
    finally:
        cv2.destroyAllWindows()
