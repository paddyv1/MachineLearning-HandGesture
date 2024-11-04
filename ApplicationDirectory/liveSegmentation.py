import cv2
import imutils
import numpy as np
import matplotlib as plt

BACKGROUND = None
import tensorflow

keras_model = tensorflow.keras.models.load_model('5_Classes2566classes', compile=True)

def run_avg(image, aWeight):
    global BACKGROUND
    #BACKGROUND VARIBALE
    if BACKGROUND is None:
        BACKGROUND = image.copy().astype("float")
        return

    # CALCULATING THE BACKGROUND IMAGES
    cv2.accumulateWeighted(image, BACKGROUND, aWeight)


def segment(image, threshold=25):
    global BACKGROUND
    #FINDING DIFF BETWEEN BACKGROUND AND CURRENT FRAME
    diff = cv2.absdiff(BACKGROUND.astype("uint8"), image)

    #THRESHOLD OF DIFFERENCE BETWEEN IMAGES
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    #CALCULATING CONTOURS
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 300, 400, 700

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        #REGION WHICH WILL BE TESTED AGAINST
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                inputModel = thresholded
                print(thresholded.shape)
                im_resized = cv2.resize(inputModel, (256, 256),interpolation=cv2.INTER_AREA)
                #print (im_resized.shape)
                cv2.imwrite("1_HSV.jpg",im_resized)
                npazz = np.expand_dims(im_resized , axis=0).reshape(256,256,1)
                #print(npazz.shape)
                img = keras_model.predict(np.expand_dims(im_resized , axis=0))
                print(img)
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()