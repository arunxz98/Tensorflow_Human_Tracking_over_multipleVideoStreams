from keras.preprocessing import image
import numpy as np


def load_image(img_path, target_size=(128,64)):
    # print(type(img_path))

    img = image.load_img(img_path, target_size=target_size,interpolation='lanczos')
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def to_tensor(frame):
    img_tensor = image.smart_resize(frame,size=(128,64))
    img_tensor = image.img_to_array(img_tensor)
    # img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor