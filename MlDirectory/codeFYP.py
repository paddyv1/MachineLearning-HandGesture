import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *

data_gen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=45,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1/255,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.2,
    interpolation_order=1,
    dtype=None
)
#C:/Users/patri/Pictures/Camera Roll/segment_training/
train_generator = data_gen.flow_from_directory("C:/Users/patri/Pictures/Camera Roll/segment_training/",
                                               target_size=(256, 256),
                                               color_mode='grayscale',
                                               classes=None,
                                               class_mode='categorical',
                                               batch_size=9,
                                               shuffle=True,
                                               seed=None,
                                               save_to_dir=None,
                                               save_prefix='',
                                               save_format='jpg',
                                               follow_links=False,
                                               subset='training',
                                               interpolation='nearest')

valid_generator = data_gen.flow_from_directory("C:/Users/patri/Pictures/Camera Roll/segment_training/",
                                               target_size=(256, 256),
                                               color_mode='grayscale',
                                               classes=None,
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=True,
                                               seed=None,
                                               save_to_dir=None,
                                               save_prefix='',
                                               save_format='jpg',
                                               follow_links=False,
                                               subset='validation',
                                               interpolation='nearest')




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
###CREATING THE CNN TO TRAIN AGAINST SEGMENTED IMAGES
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (256,256,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation = "softmax"))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])

model.fit(
        train_generator,
        validation_data=(valid_generator),
        steps_per_epoch=50,
        epochs=200)


model.save('COMPUTERVISION MODEL SEGMENT')

