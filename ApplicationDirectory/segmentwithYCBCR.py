import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.filters
#import sklearn.metrics
import cv2
import mediapipe as mp

#img = cv2.imread('C:/Users/patri/Pictures/Camera Roll/training/rock_on/rock (45).jpg')


import cv2
import numpy

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
ret, image = cam.read()

skin_min = numpy.array([0,133,77],numpy.uint8)
skin_max = numpy.array([255,173,127],numpy.uint8)
while True:
    ret, image = cam.read()

    gaussian_blur = cv2.GaussianBlur(image,(5,5),0)
    blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2YCR_CB)

#threshould using min and max values
    tre_green = cv2.inRange(blur_hsv, skin_min, skin_max)
#getting object green contour
    contours, hierarchy = cv2.findContours(tre_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#draw contours
    cv2.drawContours(image,contours,-1,(0,255,0),3)

    cv2.imshow('real', image)
    cv2.imshow('tre_green', tre_green)

    key = cv2.waitKey(10)
    if key == 27:
        break