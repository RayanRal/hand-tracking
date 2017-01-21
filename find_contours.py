import cv2
import numpy as np


def find_contours(img_hand):
    gray = cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    ret2, thresh1 = cv2.threshold(blur, 70, 220, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img_hand.shape, np.uint8)
    if thresh1 is not None:
        cv2.imshow('hand', thresh1)

    # extract the largest contour
    max_area = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

    centr = (cx, cy)
    cv2.circle(img_hand, centr, 5, [0, 0, 255], 2)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    hull = cv2.convexHull(cnt, returnPoints=False)

    if cnt.any() and hull.any():
        defects = cv2.convexityDefects(cnt, hull)
        mind = 0
        maxd = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                dist = cv2.pointPolygonTest(cnt, centr, True)
                cv2.line(img_hand, start, end, [0, 255, 0], 2)

                cv2.circle(img_hand, far, 5, [0, 0, 255], -1)
        i = 0
    return img_hand