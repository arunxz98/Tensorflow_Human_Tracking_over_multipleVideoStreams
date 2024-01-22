import numpy as np
import cv2

names = ['/home/arun/Documents/Project/videos/hotel_cam1-1.mp4',
 '/home/arun/Documents/Project/videos/hotel_cam2-1.mp4']
window_titles = ['first', 'second']

cap = [cv2.VideoCapture(i) for i in names]

frames = [None] * len(names)
gray = [None] * len(names)
ret = [None] * len(names)

while True:

    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read()


    for i,f in enumerate(frames):
        if ret[i] is True:
            # gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            cv2.imshow(window_titles[i], f)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break