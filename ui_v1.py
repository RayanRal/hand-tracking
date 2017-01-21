import cv2
import numpy as np

drawing = False  # true if mouse is pressed
mode = False  # if True, draw rectangle. Press 'm' to toggle to curve
undo = False
ix, iy = -1, -1


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img1, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img1, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img1, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv2.circle(img1, (x, y), 5, (0, 0, 255), -1)


def user_click(event, x, y, flags, param):
    global ix, iy, drawing, mode
    print (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # drawing_curve[x,y] = 1
            draw.ellipse((x, y, x + 10, y + 10), fill='blue', outline='blue')
            image.save('test.png')
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def user_click1(event, x, y, flags, param):
    global drawing, undo, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            drawing_curve.append((x, y))
        elif undo == True:
            del drawing_curve[-1]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        undo = True
    elif event == cv2.EVENT_RBUTTONUP:
        undo = False


def main():
    global drawing_curve, draw
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', user_click1)
    ret_val, img = cam.read()
    print img.shape
    drawing_curve = []
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        overlay = img.copy()
        if drawing_curve:
            for point in drawing_curve:
                cv2.circle(overlay, (point[0], point[1]), 5, (0, 0, 255), -1)

        opacity = 0.8
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

        cv2.imshow('image', img)

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()