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

# # initializing the model
model = keras.models.load_model("/content/drive/MyDrive/video_infer/saved_model/siamese_17-0.98.h5")


# giving video file inputs
names = ['/content/drive/MyDrive/video_infer/videos/hotel_cam1-2.mp4',
 '/content/drive/MyDrive/video_infer/videos/hotel_cam2-2.mp4']
window_titles = ['first', 'second']

# output file names
outputs = ['output1.avi','output2.avi']

# initializing a video writer.
def init_vidWriter(output_name):
    out = cv2.VideoWriter(
    output_name,
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

    return out


cap = [cv2.VideoCapture(i) for i in names]
writer = [init_vidWriter(i) for i in outputs]

frames = [None] * len(names)
gray = [None] * len(names)
ret = [None] * len(names)

while True:

# reading each frame from each video in a loop
    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read()
            

# making detections on frames and performing identification operation.
    for i,f in enumerate(frames):
        if ret[i] is True:
            f = cv2.resize(f, (640, 480))

            # detect people in the image
            # returns the bounding boxes for the detected objects
            boxes, weights = hog.detectMultiScale(f, winStride=(10,10))
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            persons = []
            for box in boxes:
                crop_img = f[box[1]:box[3],box[0]:box[2]]
                tensor_img = to_tensor(crop_img)
                persons.append(tensor_img)
            persons = np.array(persons)

            if not persons.shape[0] == 0:
              matches = ca.compare_and_pred(persons)
              match_boxes = [boxes[i] for i in matches]
              for (xA, yA, xB, yB) in match_boxes:
                  # overlay the detected person box in the frame
                  cv2.rectangle(f, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
            #   cv2.imshow(window_titles[i], gray[i])
              # write the frame to the video file
            # else:
            #   continue
            writer[i].write(f)
            

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


for c in cap:
    if c is not None:
        c.release()

cv2.destroyAllWindows()