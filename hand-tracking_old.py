import numpy as np
import cv2


palm_casc = cv2.CascadeClassifier('cascades/hand-haar3.xml')
# fist_casc = cv2.CascadeClassifier('fist-haar.xml')

def draw_points(img_hand, drawing_curve):
    # draw points
    overlay = img_hand.copy()
    for point in drawing_curve:
        cv2.circle(overlay, (point[0], point[1]), 8, (0, 0, 255), -1, lineType=4)
    opacity = 0.8
    cv2.addWeighted(overlay, opacity, img_hand, 1 - opacity, 0, img_hand)


cap = cv2.VideoCapture(0)
init = False
while not init:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img2', frame)
    cv2.waitKey(10) & 0xff
    palms = palm_casc.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=25,
        minSize=(100, 100),
        flags=2
    )
    if type(palms) is np.ndarray:
        points = palms[0].tolist()
        img = frame
        init = True

drawing_curve = []
# minx, miny, maxx, maxy = points[0], points[1], points[2], points[3]
x, y, w, h = points[0], points[1], points[2], points[3]
# r, h, c, w = minx, miny, maxx-minx, maxy-miny
# (xmin, ymin, xmax - xmin, ymax - ymin)
# track_window = (c, r, w, h)
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = img[x:x+w, y:y+h]
# roi = img[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either iterations or move by at least some pt
term_crit = (cv2. TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)

iter = 0
nf = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    if ret:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        (c, r, w, h) = track_window
        img_hand = img[r:r + h, c:c + w]
        # img_hand, finger_coords = find_contours.find_contours(img_hand)
        # img_hand, finger_coords = hd.draw_final(img_hand)
        # drawing_curve.append((finger_coords[0], finger_coords[1]))
        # cv2.imshow('hand', img_hand)

        img[r:r + h, c:c + w] = img_hand

        # Draw it on image - meanShift
        x, y, w, h = track_window
        drawing_curve.append((x+(w/2), y+(h/2)))
        draw_points(img, drawing_curve)
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)

        if iter % 10 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            palms = palm_casc.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=9,
                minSize=(100, 100),
                maxSize=(600, 600),
                flags=2
            )
            if type(palms) is not np.ndarray:
                nf += 1
            else:
                nf = 0
            if nf == 7:
                drawing_curve = []

        # Draw it on image - CamShift
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        # cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)

        cv2.imshow('img2', img)

        k = cv2.waitKey(30) & 0xff
        iter += 1
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img)

    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()