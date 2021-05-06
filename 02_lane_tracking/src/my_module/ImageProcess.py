import cv2
import numpy as np

class ImageProcessing(object):

    def __init__(self):
        pass


    def bgr2gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def bgr2blur(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)


    def img2canny(self, img, low_th, high_th):
        return cv2.Canny(img, low_th, high_th)


    def img2hls(self, img, low_th, high_th):
        _, L, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
        _, lane_img = cv2.threshold(L, low_th, high_th, cv2.THRESH_BINARY)
        return lane_img


    def set_roi(self, img):
        height, width = img.shape[0:2]

        region = np.array([[
            (0, height),
            (240, height-200),
            (width-240, height-200),
            (width, height)
        ]])
        
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, region, 255)
        return cv2.bitwise_and(img, mask)


    def perspective_img(self, img):
        height, width = img.shape[0:2]

        point_1 = [260, height-200]
        point_2 = [30, height]
        point_3 = [width-260, height-200]
        point_4 = [width-30, height]

        warp_src  = np.array([point_1, point_2, point_3, point_4], dtype=np.float32)
        warp_dist = np.array([[0, 0],\
                              [0, height],\
                              [width, 0],\
                              [width, height]],\
                             dtype=np.float32)

        M = cv2.getPerspectiveTransform(warp_src, warp_dist)
        Minv = cv2.getPerspectiveTransform(warp_dist, warp_src)
        warp_img = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
        return warp_img, M, Minv

    
    def rev_perspective_img(self, img, Minv):
        height, width = img.shape[0:2]
        return cv2.warpPerspective(img, Minv, (width, height))


    def calibrate_img(self, img):
        height, width = img.shape[0:2]

        mtx = np.array([[422.037858, 0.0, 245.895397],\
                        [0.0, 435.589734, 163.625535],\
                        [0.0, 0.0, 1.0]])
        dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
        cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

        tf_img = cv2.undistort(img, mtx, dist, None, cal_mtx)
        x, y, w, h = cal_roi
        tf_img = tf_img[y:y+h, x:x+w]

        return cv2.resize(tf_img, (width, height))
