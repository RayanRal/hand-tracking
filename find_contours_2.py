import cv2
import numpy as np
import image_analysis


class HandDetection:
    def __init__(self):
        self.trained_hand = False
        self.hand_row_nw = None
        self.hand_row_se = None
        self.hand_col_nw = None
        self.hand_col_se = None
        self.hand_hist = None

    def draw_hand_rect(self, frame):
        rows, cols, _ = frame.shape

        self.hand_row_nw = np.array([6 * rows / 20, 6 * rows / 20, 6 * rows / 20,
                                     10 * rows / 20, 10 * rows / 20, 10 * rows / 20,
                                     14 * rows / 20, 14 * rows / 20, 14 * rows / 20])

        self.hand_col_nw = np.array([9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
                                     9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
                                     9 * cols / 20, 10 * cols / 20, 11 * cols / 20])

        self.hand_row_se = self.hand_row_nw + 10
        self.hand_col_se = self.hand_col_nw + 10

        size = self.hand_row_nw.size
        for i in xrange(size):
            cv2.rectangle(frame, (self.hand_col_nw[i], self.hand_row_nw[i]), (self.hand_col_se[i], self.hand_row_se[i]),
                          (0, 255, 0), 1)
        black = np.zeros(frame.shape, dtype=frame.dtype)
        frame_final = np.vstack([black, frame])
        return frame_final

    def train_hand(self, frame):
        self.set_hand_hist(frame)
        self.trained_hand = True

    def set_hand_hist(self, frame):
        # TODO use constants, only do HSV for ROI
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([90, 10, 3], dtype=hsv.dtype)

        size = self.hand_row_nw.size
        for i in xrange(size):
            roi[i * 10:i * 10 + 10, 0:10] = hsv[self.hand_row_nw[i]:self.hand_row_nw[i] + 10,
                                            self.hand_col_nw[i]:self.hand_col_nw[i] + 10]

        self.hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.hand_hist, self.hand_hist, 0, 255, cv2.NORM_MINMAX)

    def apply_hist_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hand_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 100, 255, 0)
        thresh = cv2.merge((thresh, thresh, thresh))

        cv2.GaussianBlur(dst, (3, 3), 0, dst)

        res = cv2.bitwise_and(frame, thresh)
        return res

    def draw_final(self, frame):
        hand_masked = image_analysis.apply_hist_mask(frame, self.hand_hist)

        contours = image_analysis.contours(hand_masked)
        if contours is not None and len(contours) > 0:
            max_contour = image_analysis.max_contour(contours)
            hull = image_analysis.hull(max_contour)
            centroid = image_analysis.centroid(max_contour)
            cv2.circle(hand_masked, centroid, 5, [0, 255, 0], -1)
            defects = image_analysis.defects(max_contour)

            if centroid is not None and defects is not None and len(defects) > 0:
                farthest_point = image_analysis.farthest_point(defects, max_contour, centroid)

                if farthest_point is not None:
                    cv2.circle(hand_masked, farthest_point, 5, [0, 0, 255], -1)

        cv2.imshow('hand', hand_masked)
        return frame, centroid
