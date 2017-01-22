import cv2

img = cv2.imread('hand_fist.jpg')
# img = cv2.imread('hand_palm.jpg')
# palm_casc = cv2.CascadeClassifier('palm-haar.xml')
palm_casc = cv2.CascadeClassifier('cascades/hand-haar3.xml')
height, width = 240, 320
# res = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# def HaarDetectObjects(image, cascade, storage, scale_factor=None, min_neighbors=None, flags=None, min_size=None): # real signature unknown; restored from __doc__
# palms = cv2.cv.HaarDetectObjects(gray, palm_casc)
palms = palm_casc.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=25,
    minSize=(50, 50) #,
    # maxSize=(120, 120)
)
print palms

for (x, y, w, h) in palms:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('hand_fist-mod.jpg', gray)
