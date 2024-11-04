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

im = cv2.imread('dataset/new_peace_1.jpg')
im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY))
plt.show()

#Open a simple image
#img=cv2.imread("dataset/new_peace.jpg")
#ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

#plt.imshow(ycbcr_img)
#plt.show()

import cv2
import matplotlib.pyplot as plt
#"C:\Users\patri\Pictures\Camera Roll\training\fist\fist (1).jpg"
# load the input image
img = cv2.imread('C:/Users/patri/Pictures/Camera Roll/training/fist/fist (1).jpg')

# convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
ret,thresh1 = cv2.threshold(gray,70,255,0)
thresh2 = cv2.bitwise_not(thresh1)
# convert BGR to RGB to display using matplotlib
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display Original, Grayscale and Binary Images
plt.subplot(131),plt.imshow(imgRGB,cmap = 'gray'),plt.title('Original Image'), plt.axis('off')
plt.subplot(132),plt.imshow(gray,cmap = 'gray'),plt.title('Grayscale Image'),plt.axis('off')
plt.subplot(133),plt.imshow(thresh2,cmap = 'gray'),plt.title('Binary Image'),plt.axis('off')
plt.show()