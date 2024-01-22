import numpy as np
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()  #This line makes the GUI interactable without having to call waitkey


names = ['/home/arun/Documents/Project/videos/hotel_cam1-1.mp4',
 '/home/arun/Documents/Project/videos/hotel_cam2-1.mp4']
window_titles = ['first', 'second']

cap = [cv2.VideoCapture(i) for i in names]

frames = [None] * len(names);
gray = [None] * len(names);
ret = [None] * len(names);

while True:

    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read();


    for i,f in enumerate(frames):
        if ret[i] is True:
            f = cv2.resize(f, (640, 480))
            boxes, weights = hog.detectMultiScale(f, winStride=(8,8))
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            for (xA, yA, xB, yB) in boxes:
                  # overlay the detected person box in the frame
                  cv2.rectangle(f, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
            cv2.imshow(window_titles[i], f)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break