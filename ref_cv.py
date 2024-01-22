from ast import Global
from tkinter import W
import numpy as np
import cv2
from siamese_network.imageutils import to_tensor

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture("/home/arun/Documents/Project/videos/mixkit-subway-network-in-tokyo-4453.mp4")

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False : print("video not found please provide exact path for valid video")

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # print(boxes)
    frames = []
    for box in boxes:
        crop_img = frame[box[1]:box[3],box[0]:box[2]]
        tensor_img = to_tensor(crop_img)
        frames.append(tensor_img)
    frames = np.array(frames)
    print(frames.shape)
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
