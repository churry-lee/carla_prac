import cv2
import numpy as np

class HoughDetect(object):
    
    def __init__(self):
        pass
    

    def houghtf(self, img, th, minL, maxL):
        return cv2.HoughLinesP(img, 1, np.pi/180, th, np.array([]), minL, maxL)


    def calculate_lines(self, img, lines):
        left, right = [], []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                #print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
                parameters = np.array([0, 0])
                if x1 != x2:
                    parameters = np.polyfit((x1, x2), (y1, y2), 1)
                    # print("slope: {}".format(parameters[0]))
                else:
                    continue
                slope = parameters[0]
                y_intercept = parameters[1]
                if x2 < 320:
                    left.append((slope, y_intercept))
                elif 320 < x2:
                    right.append((slope, y_intercept))
        else:
            print("No lines in image!")

        if len(left) != 0:
            left_avg = np.average(left, axis=0)
            left_line = self.calculate_coordinates(img, left_avg)
        elif len(left) == 0:
            left_line = np.array([0, 0, 0, 0])

        if len(right) != 0:
            right_avg = np.average(right, axis=0)
            right_line = self.calculate_coordinates(img, right_avg)
        elif len(right) == 0:
            right_line = np.array([0, 0, 0, 0])

        return np.array([left_line, right_line])


    def calculate_coordinates(self, img, parameters):
        height, width = img.shape[0:2]

        slope, intercept = parameters
        
        y1 = height
        y2 = int(y1 - 150)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])


    def visualize_lines(self, img, lines):
        try:
            if lines is not None:
                for x1, y1, x2, y2 in lines:
                    if x1 > 320:  # Draw right line
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                    else:         # Draw left line
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
            else:
                print('No lines in image!')
            return img
        except OverflowError:
            print("Rais OverFlowError")
        except TypeError:
            print("Rais TypeError")
