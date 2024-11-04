from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
import keras
import os
import pandas as pd
import tensorflow

keras_model = tensorflow.keras.models.load_model('5_Classes', compile=True)

from PIL import Image
import numpy as np
import skimage

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = skimage.transform.resize(np_image, (256, 256, 1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('../SimpleCNN/seg_pointer (201).jpg')
predicitons = keras_model.predict(image)
print (predicitons)
predictionL = predicitons[0]
max_value = max(predictionL)
max_index = np.where(predictionL == max_value)
print("Maximum Index position: ",max_index)
image = load('../SimpleCNN/seg_fist (1).jpg')
predicitons = keras_model.predict(image)
print (predicitons)
predictionL = predicitons[0]
max_value = max(predictionL)
max_index = np.where(predictionL == max_value)
print("Maximum Index position: ",max_index)

image = load('../SimpleCNN/1_HSV.jpg')
predicitons = keras_model.predict(image)
print (predicitons)
predictionL = predicitons[0]
max_value = max(predictionL)
max_index = np.where(predictionL == max_value)
print("Maximum Index position: ",max_index)
