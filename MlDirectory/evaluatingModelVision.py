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
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = data_gen.flow_from_directory("C:/Users/patri/Pictures/Camera Roll/segment_training/",
                                               target_size=(256, 256),
                                               color_mode='grayscale',
                                               classes=None,
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=False,
                                               seed=None,
                                               save_to_dir=None,
                                               save_prefix='',
                                               save_format='jpg',
                                               follow_links=False,
                                               subset='validation',
                                               interpolation='nearest')



from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = keras.models.load_model("fyp test")

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['Fist', 'Palm', 'Peace', 'Pointer', 'Rock', 'three']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))