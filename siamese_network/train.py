# importing framework libraries
from statistics import mode
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from keras import optimizers
import tensorflow as tf
import tensorboard

# importing custom libraries
import data
from model import EfficientNet_siamese_model

# setting hyperparameters and configs
train_batch_size = 2
epochs = 3
learning_rate = 0.001
input_shape = (128,64,3)
checkpoint_path = "./checkpoints/siamese_{epoch:02d}-{val_accuracy:.2f}.h5"
tensorboard_dir = "./tensorboard_logs"
save_dir = "./saved_model/siamese.h5"


# loading and preprocessing data
_data = data.Data()
x_train,y_train = _data.getDataSet("/home/arun/Documents/Project/archive/Market-1501-v15.09.15/query")
x_test,y_test = _data.getDataSet("/home/arun/Documents/Project/archive/Market-1501-v15.09.15/query")

train_datagen = _data.pairs_data_generator(x_train,y_train,batch_size=train_batch_size)
test_datagen = _data.pairs_data_generator(x_test,y_test,batch_size=2)


# loading the model architecture which is defined in model.py
network = EfficientNet_siamese_model()
model = network.siamese_model()
# model.summary()

# running prerequisites for model before fit
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,loss="binary_crossentropy",metrics=['accuracy'])
model_callbacks = [
    ModelCheckpoint(checkpoint_path,monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1),
    TensorBoard(log_dir=tensorboard_dir)
]

# training the model
model.fit(
    train_datagen,
    validation_data=test_datagen,
    steps_per_epoch=2,
    epochs=epochs,
    callbacks=model_callbacks,
    validation_steps=1)

# saving the model
model.save(filepath=save_dir)