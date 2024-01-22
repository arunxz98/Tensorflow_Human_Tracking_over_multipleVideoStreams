import keras
import glob
import numpy as np
import tensorflow as tf
import efficientnet.keras
import time
import numpy as np
import cv2

import imageutils


ref_person = False
# initializing the model
model = keras.models.load_model("/content/drive/MyDrive/outsourcing/video_infer/saved_model/siamese_17-0.98.h5")


def copy_frames(frame,size:int):
    probe_frames = []
    for _ in range(size):
        probe_frames.append(frame.copy())

    return np.asarray(probe_frames)

def get_ref_person():
  ref_person = imageutils.load_image("./ref_img/1.png")
  return ref_person

def compare_and_pred(frames,ref = True):

  global ref_person
  if not ref_person:
    if ref == True:
      ref_person = get_ref_person()
      cv2.imwrite(f"./ref_taken/ref_{time.time()}.png",ref_person)
    else:
      ref_person = frames[0]
      cv2.imwrite(f"./ref_taken/ref_{time.time()}.png",ref_person)

  ref_set = copy_frames(ref_person,frames.shape[0])
  pred = model.predict([frames,ref_set])
  matches = np.where(pred>=0.80)
  return matches[0]
