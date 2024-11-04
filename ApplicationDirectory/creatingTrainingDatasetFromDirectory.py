import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
from skimage import transform
import io
import os
import numpy as np
from PIL import Image, ImageEnhance
main_dir = "C:/Users/patri/Pictures/Camera Roll/training/"
new_dir = "C:/Users/patri/Pictures/Camera Roll/segment_training/"
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    print(subdir_path)

    os.makedirs(new_dir + "segment_" + subdir)
    if os.path.isdir(subdir_path):
        
        print(f"Processing images in {subdir} folder:")

        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):

            source = (f'{subdir_path}/{filename}')
            newsource = (f'{new_dir}segment_{subdir}/seg_{filename}')
            img = cv2.imread(source)

            # convert the input image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # apply thresholding to convert grayscale to binary image
            ret, thresh1 = cv2.threshold(gray, 70, 255, 0)
            thresh2 = cv2.bitwise_not(thresh1)
            # convert BGR to RGB to display using matplotlib
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            thresh3 = cv2.resize(thresh2, (256,256))
            cv2.imwrite(newsource, thresh3)




