from json import load
import keras
import glob
import numpy as np
import tensorflow as tf
import efficientnet.keras
import time

from imageutils import load_image
from losses import cosine_distance

def load_model(model_path:str):
    model = keras.models.load_model(model_path)
    return model


# images_list = glob.glob()
# Change the path according to your local path settings

img1 = load_image("/home/arun/Documents/Project/archive/Market-1501-v15.09.15/query/0001_c4s6_000810_00.jpg")
img2 = load_image("/home/arun/Documents/Project/archive/Market-1501-v15.09.15/query/0003_c3s3_064744_00.jpg")
img3 = load_image("/home/arun/Documents/Project/archive/Market-1501-v15.09.15/query/0004_c1s6_016996_00.jpg")
img4 = load_image("/home/arun/Documents/Project/archive/Market-1501-v15.09.15/query/0004_c3s3_065619_00.jpg")


model = keras.models.load_model(
    "/home/arun/Documents/Project/code/saved_model/siamese_17-0.98.h5",
)

con1 = []
con2 = []

for _ in range(10):
    con1.append(img1)
    con2.append(img2)

# print(np.array(con1).squeeze().shape)
start = time.time()
pred = model.predict([np.array(con1).squeeze(),np.array(con2).squeeze()])
end = time.time()


print(pred)
print(f"latency = {end - start}")