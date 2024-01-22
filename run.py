from re import I
import numpy as np
import cv2
import keras
import glob
import efficientnet.keras
import time

from imageutils import to_tensor
import compare_algo as ca

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()  #This line makes the GUI interactable without having to call waitkey

# initializing the model
model = keras.models.load_model("/content/drive/MyDrive/video_infer/saved_model/siamese_17-0.98.h5")

# open webcam video stream
cap = cv2.VideoCapture("/content/drive/MyDrive/video_infer/videos/mixkit-subway-network-in-tokyo-4453.mp4")

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)  
    if ret == False : print("video not found please provide full path for valid video")

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
    
    matches = ca.compare_and_pred(frames)
    match_boxes = [boxes[i] for i in matches]
    # for (xA, yA, xB, yB) in match_boxes:
    #   print(xA, yA, xB, yB)
    for (xA, yA, xB, yB) in match_boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break