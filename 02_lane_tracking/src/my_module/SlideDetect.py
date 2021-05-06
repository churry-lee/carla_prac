import cv2
import numpy as np
import matplotlib.pyplot as plt

class SlideDetect(object):
    
    def __init__(self):
        pass

    def histogram(self, img):
        histo = np.sum(img[img.shape[0]//2: , :], axis=0)
        midpoint = np.int(histo.shape[0]//2)
        leftx_current = np.argmax(histo[:midpoint])
        rightx_current = np.argmax(histo[midpoint:]) + midpoint
        #plt.plot(histo)
        #plt.show()
        return leftx_current, rightx_current

    def SlidingWindow(self, img, leftx_current, rightx_current, w_count):
        height, width = img.shape[0:2]
        window_height = np.int(height//w_count)
        nz = img.nonzero()

        left_lane = []
        right_lane = []

        lx, ly, rx, ry = [], [], [], []
        out_img = np.dstack((img, img, img))
        #out_img = np.dstack((img, img, img)) * 255

        margin = 80

        for window in range(w_count):
            win_yl = height - (window+1) * window_height
            win_yh = height - (window*window_height)

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            left_window = cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (255, 0, 0), 2)
            right_window = cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 0, 255), 2)

            good_left_inds = ((nz[0]>=win_yl)&(nz[0]<win_yh)&(nz[1]>=win_xll)&(nz[1]<win_xlh)).nonzero()[0]
            good_right_inds = ((nz[0]>=win_yl)&(nz[0]<win_yh)&(nz[1]>=win_xrl)&(nz[1]<win_xrh)).nonzero()[0]

            left_lane.append(good_left_inds)
            right_lane.append(good_right_inds)

            if len(good_left_inds) > 100:
                leftx_current = np.int(np.mean(nz[1][good_left_inds]))
            if len(good_right_inds) > 100:
                rightx_current = np.int(np.mean(nz[1][good_right_inds]))

            lx.append(leftx_current)
            ly.append((win_yl + win_yh)//2)

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)//2)


        self.left_lane = np.concatenate(left_lane)
        self.right_lane = np.concatenate(right_lane)

        self.lfit = np.polyfit(np.array(ly), np.array(lx), 2)
        self.rfit = np.polyfit(np.array(ry), np.array(rx), 2)

        out_img[nz[0][self.left_lane], nz[1][self.left_lane]] = [255, 0, 0]
        out_img[nz[0][self.right_lane] , nz[1][self.right_lane]] = [0, 0, 255]
        return out_img

    def draw_lane(self, img, warp_img, Minv):
        height, width = img.shape[0:2]

        yMax = warp_img.shape[0]
        ploty = np.linspace(0, yMax, yMax, endpoint=False)
        color_warp = np.zeros_like(warp_img).astype(np.uint8)

        left_fitx = self.lfit[0]*ploty**2 + self.lfit[1]*ploty + self.lfit[2]
        right_fitx = self.rfit[0]*ploty**2 + self.rfit[1]*ploty + self.rfit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
        #print("mean_x: {}".format(mean_x))
        #print("pts_mean: {}".format(pts_mean))

        color_warp = cv2.fillPoly(warp_img, np.int_([pts]), (10, 30, 100))
        inv_warp = cv2.warpPerspective(color_warp, Minv, (width, height))

        return cv2.addWeighted(img, 1.0, inv_warp, 1.0, 0)
        #return cv2.bitwise_and(img, inv_warp)
