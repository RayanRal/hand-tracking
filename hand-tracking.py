import numpy as np
import cv2
import get_points
import find_contours
import find_contours_2 as fc2


def draw_points(img_hand, drawing_curve):
    # draw points
    overlay = img_hand.copy()
    for point in drawing_curve:
        cv2.circle(overlay, (point[0], point[1]), 5, (0, 0, 255), -1)
    opacity = 0.8
    cv2.addWeighted(overlay, opacity, img_hand, 1 - opacity, 0, img_hand)


cap = cv2.VideoCapture(0)
# MIL, TLD, KCF, BOOSTING, MEDIANFLOW
tracker = cv2.Tracker_create("KCF")
tracker_initialized = False
points, img = get_points.find_points(cap)

if not points:
    print "ERROR: No object to be tracked."
    exit()

bbplus = 50
drawing_curve = []
minx, miny, maxx, maxy = points[0][0], points[0][1], points[0][2], points[0][3]
r, h, c, w = minx, miny, maxx-minx, maxy-miny
# (xmin, ymin, xmax - xmin, ymax - ymin)
# track_window = (c, r, w, h)
track_window = (minx, miny, maxx-minx, maxy-miny)

# set up the ROI for tracking
# roi = img[r:r + h, c:c + w]
# hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# hd = fc2.HandDetection()
# hd.draw_hand_rect(roi)
# hd.set_hand_hist(roi)

# Setup the termination criteria, either iterations or move by at least some pt
# term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    if ret:
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        if not tracker_initialized:
            ok = tracker.init(frame, track_window)
            tracker_initialized = True

        ok, track_window = tracker.update(frame)

        # (c, r, w, h) = track_window
        # img_hand = img[r-bbplus:r + h + bbplus, c:c + w + bbplus]
        # img_hand, finger_coords = find_contours.find_contours(img_hand)
        # img_hand, finger_coords = hd.draw_final(img_hand)
        # drawing_curve.append((finger_coords[0], finger_coords[1]))
        # drawing_curve.append((15, 15))
        # cv2.imshow('hand', img_hand)

        # draw_points(img_hand, drawing_curve)

        # img[r-bbplus:r + h + bbplus, c:c + w + bbplus] = img_hand

        # Draw it on image - meanShift
        x, y, w, h = track_window
        # x, y, w, h = int(x), int(y), int(w), int(h)
        # cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
        if ok:
            p1 = (int(track_window[0]), int(track_window[1]))
            p2 = (int(track_window[0] + track_window[2]), int(track_window[1] + track_window[3]))
            cv2.rectangle(img, p1, p2, (200, 0, 0), 2)

        # Draw it on image - CamShift
        # pts = cv2.cv.BoxPoints(ret)
        # pts = np.int0(pts)
        # cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)

        cv2.imshow('img2', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img)

    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()