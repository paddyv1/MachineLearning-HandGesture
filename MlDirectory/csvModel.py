import csv

import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

RANDOM_SEED = 35

dataset = 'keypointspotifyfinal.csv'
model_save_path = 'fyp test'



X_keypoint = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_keypoint = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X_keypoint, y_keypoint, train_size=0.75, random_state=RANDOM_SEED)

###model used to train against the location of coords
model = Sequential()

model.add(Input(shape=(42)))  #42
model.add(Dropout(0.27))  #0.25
model.add(Dense(22, activation = "relu"))  #20
model.add(Dropout(0.38))  #0.4
model.add(Dense(12, activation = "relu"))  #10
model.add(Dense(7, activation='softmax'))  #7

model.summary()




###saving models after each epoch because training will end early once no more accuracy canb be acheived
model_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)

##implementing early stopping, learnt about in 2 years of studying ai
stopping_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
###compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#print (X_train[1])
#print (X_train[1].shape)
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[model_callback, stopping_callback]
)


val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)


model = tf.keras.models.load_model(model_save_path)

predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report





Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report')

print(classification_report(y_test, y_pred))
#print_confusion_matrix(y_test, y_pred)