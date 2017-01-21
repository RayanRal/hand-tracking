import numpy as np
import cv2
import get_points
import find_contours


cap = cv2.VideoCapture(0)
points, img = get_points.find_points(cap)

if not points:
    print "ERROR: No object to be tracked."
    exit()

bbplus = 100
minx, miny, maxx, maxy = points[0][0], points[0][1], points[0][2], points[0][3]
r, h, c, w = minx, miny, maxx-minx, maxy-miny
# (xmin, ymin, xmax - xmin, ymax - ymin)
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = img[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either iterations or move by at least some pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    if ret:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        (c, r, w, h) = track_window
        img_hand = img[r-bbplus:r + h + bbplus, c-bbplus:c + w + bbplus]
        img_hand = find_contours.find_contours(img_hand)

        # Draw it on image - meanShift
        # x, y, w, h = track_window
        # cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
        # Draw it on image - CamShift
        img[r-bbplus:r + h + bbplus, c-bbplus:c + w + bbplus] = img_hand
        pts = cv2.cv.BoxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)
        cv2.imshow('img2', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img)

    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()