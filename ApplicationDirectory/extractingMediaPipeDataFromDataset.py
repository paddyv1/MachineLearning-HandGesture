import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import os
import numpy as np
from skimage import transform
import csv
import itertools



import os
import numpy as np
from PIL import Image, ImageEnhance
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
main_dir = "C:/Users/patri/Pictures/Camera Roll/training/"
#C:/Users/patri/Pictures/Camera Roll/training/

directoryCounter = 0
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    print(subdir_path)


    if os.path.isdir(subdir_path):
        count = 0  # counter to keep track of number of images processed
        print(f"Processing images in {subdir} folder:")

        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):

            source = (f'{subdir_path}/{filename}')

            img = cv2.imread(source)


            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            arrayPoints = []
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    #print(handLms)
                    baseCoords = [0, 0]
                    #print(baseCoords)
                    for id, lm in enumerate(handLms.landmark):
                        # print(id, lm)
                        height, width, c = img.shape
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        if id == 0:
                            baseCoords = [int(cx), int(cy)]
                            arrayPoints.append([0, 0])
                        else:
                            xdiff = baseCoords[0] - cx
                            ydiff = baseCoords[1] - cy
                            arrayPoints.append([xdiff, ydiff])


            # FPS Counter

            #print(arrayPoints)

                arrayPointsList = list(
                    itertools.chain.from_iterable(arrayPoints))
                #print(arrayPointsList)

                max_value = max(list(map(abs, arrayPointsList)))


                def MAX_DISTANCE_(n):
                    return n / max_value


                arrayPointsList = list(map(MAX_DISTANCE_, arrayPointsList))
                arrayPointsList.insert(0, directoryCounter)

                #print(arrayPointsList)

                csv_path = 'keypointspotifyfinal.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([*arrayPointsList])
    directoryCounter = directoryCounter + 1