import keras
from efficientnet.tfkeras import EfficientNetB0
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K


# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

from losses import cosine_distance, euclidean_distance, distance_output_shape

class EfficientNet_siamese_model:
    
    def __init__(self,input_shape:tuple = (128,64,3)):
        self.input_shape = input_shape

    def base_model(self):
        # importing efficientNetB0 as a base model.
        efficientnetModel = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        outputs1 = efficientnetModel.output
        gap = keras.layers.GlobalAveragePooling2D()(outputs1)
        signatures = keras.layers.Dense(2048, activation = "sigmoid")(gap)

        return Model(efficientnetModel.input, signatures)

    def siamese_model(self):
        
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        effnet_base_model = self.base_model()

        embedding1 = effnet_base_model(input_a)
        embedding2 = effnet_base_model(input_b)

        distance = Lambda(cosine_distance, 
                  output_shape=distance_output_shape)([embedding1, embedding2])

        model = Model([input_a,input_b],distance)
        
        return model

if __name__ == "__main__":
    network = EfficientNet_siamese_model()
    siamese = network.siamese_model()
    siamese.summary()