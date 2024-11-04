import customtkinter as ctk
import os
import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import time
import tensorflow
import itertools
import numpy
import mouse
from threading import Thread
import pyautogui
import keyboard
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
from dotenv import load_dotenv
import os
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session
import requests
import json
from PIL import Image, ImageTk

keras_model_hands = tensorflow.keras.models.load_model('Final_Model_For_General_Control', compile=True)
keras_model_spotify = tensorflow.keras.models.load_model('Final_Model_For_Spotify_Control', compile=True)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
wScreen, hScreen = pyautogui.size()


###FUCNTION TO OPEN INSTRUCTION MANUAL FOR THE APPLICATION
def open_win():
    ###SETTING VARIABLES FOR NEW TKINTER WINDOW
   new= ctk.CTkToplevel(root)
   new.geometry("750x500")
   new.title("User Manual")
   #Create a Label in New window
   title_instructions = ctk.CTkLabel(new, text="Instructions", font=ctk.CTkFont(size=30, weight="bold"))
   title_instructions.pack(padx=10, pady=(40, 20))


   width = root.winfo_screenwidth()
   height = root.winfo_screenheight()

   #adding scrollable frame
   frame = ctk.CTkScrollableFrame(new, width=width, height=height)

   #adding gesture Instructions
   label = ctk.CTkLabel(frame, text='Global Instructions:\nDuring anytime that a gesture capturing window is open press q to close it.\n\n\n\n\n')
   label.pack()

   label = ctk.CTkLabel(frame,text='Gestures and Their Controls for Computer Control.',font=ctk.CTkFont(size=30, weight="bold"))
   label.pack()

###ADDING IMAGES WITH DESCRIPTION TO SCROLLABLE FRAME IN INSTRUCTION WINDOW
   im = Image.open('tkinterimages/pc/fist (1).jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame,text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Fist gesture is neutral and has nothing, could be assigned a role at a later date.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/palm (1).jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,
                        text='The Palm image is to perform a single left click.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/peace (1).jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,
                        text='The Peace gesture is to perform a double left click. \n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/pointer (1).jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,
                        text='The Pointer gesture is to control the mouse.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/rock (1).jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,text='The Rock gesture is neutral and can be assigned at a later date.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/three.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,text='The Three gesture is to perfrom a right click.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/thumb_down.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,text='The Thumb Down gesture to mouse wheel scroll down.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/pc/thumb_up.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame,text='The Thumb Up gesture to mouse wheel scroll up.\n\n\n\n\n')
   label.pack()




   label = ctk.CTkLabel(frame, text='\n\n\nGestures and Their Controls for Spotify Control.\n\n\n',font=ctk.CTkFont(size=30, weight="bold"))
   label.pack()

   im = Image.open('tkinterimages/spotify/back.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Back gesture is to go to the previous song.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/spotify/next.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Next gesture is to go to the next song.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/spotify/pause.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Pause gesture is to pause the current song.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/spotify/play.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Play gesture is to resume playing the current song.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/spotify/voldown.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Volume Down gesture is to reduce the volume of the song.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/spotify/volup.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Volume Up gesture is to increase the volume of the song.\n\n\n\n\n')
   label.pack()

   im = Image.open('tkinterimages/spotify/zneutral.jpg')
   resized_image = im.resize((300, 205), Image.LANCZOS)
   ph = ImageTk.PhotoImage(resized_image)
   label = ctk.CTkLabel(frame, text=None, image=ph)
   label.pack()
   label = ctk.CTkLabel(frame, text='The Neutral gesture is currently not assigned a role and could be in the future.\n\n\n\n\n')
   label.pack()

   frame.pack()




###FUNCTION TO MOVE MOUSE
def move_mouse(xpos, ypos):
    mouse.move(xpos*2.5, ypos*2.5, duration =0.01)


###FUCNTION TO START CAPTURING AND PROCESS GESTURES FOR GENERAL
###COMPUTER CONTROL
def start_capturing():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)


    previousTime = 0
    currentTime = 0
    previousCoordx, previousCoordy = 0, 0
    currentX = 0
    currentY = 0

    global counterLeftClick
    global counterDoubleLeft
    global counterRight


    counterLeftClick = 20
    counterDoubleLeft = 20
    counterRight = 20

    while True:
        success, img = cap.read()
        flip_img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
        ###PROCESSING THE IMAGES FRAMES CAPTURED BY THE WEBCAM
        results = hands.process(imgRGB)

        arrayPoints = []
        ###CHECKING IF A HAND IS PRESENT IN THE FRAME
        if results.multi_hand_landmarks:
            ###ITERATRING THROUGH ALL OF THE LANDMARKS OF THE HAND GESTURE CAPTURED
            for handLms in results.multi_hand_landmarks:
                baseCoords = [0, 0]
                # print(baseCoords)
                for id, lm in enumerate(handLms.landmark):

                    height, width, c = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    ###GETTING POSITION OF POINTER FINGER FOR MOUSE CONTROL
                    if id == 8:
                        currentX = cx
                        currentY = cy
                    ###GETTING COORDINATES FOR WRIST LOCATION WHICH IS THE BASE SET OF COORDS
                    if id == 0:
                        baseCoords = [int(cx), int(cy)]
                        arrayPoints.append([0, 0])
                    else:
                        xdiff = baseCoords[0] - cx
                        ydiff = baseCoords[1] - cy
                        arrayPoints.append([xdiff, ydiff])
                ###DRAWING THE SKELETON ONTO THE FRAME
                mpDraw.draw_landmarks(flip_img, handLms, mpHands.HAND_CONNECTIONS)

            arrayPointsList = list(
                itertools.chain.from_iterable(arrayPoints))
            # print(arrayPointsList)

            MAX_DISTANCE = max(list(map(abs, arrayPointsList)))

            def CALCULATING_DISTANCE_(n):
                return n / MAX_DISTANCE

            arrayPointsList = list(map(CALCULATING_DISTANCE_, arrayPointsList))
            arrayPointsList = numpy.asarray(arrayPointsList)


            try:
                ###PASSING PREDICITON INTO CORRECT INPUT SHAPE
                ###THEN PASSING THYE NUMPY ARRAY INTO THE MODEL SO THAT IT CAN BE PREDICTED AGAINST
                arrayPointsList = arrayPointsList.reshape(1, 42)
                prediction = keras_model_hands.predict(arrayPointsList, verbose=0)
                prediction = prediction.tolist()
                gesture = prediction[0].index(max(prediction[0]))

            except:
                command = None

                #print("error")

            ###OUTPUTTING TO THE USER WHAT GESTURE IS DETECTED
            ###PERFORMING THE APPROPRIATE ACTION DETERMINED BY THE GESTURE
            #print(gesture)
            if gesture == 0:
                command = "Fist"
                previousCoordx = 0
                previousCoordy = 0

            elif gesture == 1:
                command = "Palm"
                if counterDoubleLeft > 0:
                    counterDoubleLeft = counterDoubleLeft - 1
                else:
                    mouse.double_click(button='left')
                    counterDoubleLeft = 40



            elif gesture == 2:
                command = "Peace"
                if counterLeftClick > 0:
                    counterLeftClick = counterLeftClick -1
                else:
                    mouse.click(button='left')
                    counterLeftClick = 40



            elif gesture == 3:
                command = "Pointer"
                if previousCoordx == 0 and previousCoordy == 0:
                    previousCoordx = currentX
                    previousCoordy = currentY

                move_mouse(currentX, currentY)



            elif gesture == 4:
                command = "Rock"
                previousCoordx = 0
                previousCoordy = 0

            elif gesture == 5:
                command = "Three"
                if counterRight > 0:
                    counterRight = counterRight - 1
                else:
                    mouse.click(button='right')
                    counterRight = 40



            elif gesture == 6:
                command = "thumb_down"
                previousCoordx = 0
                previousCoordy = 0
                mouse.wheel(delta=-2)

            elif gesture == 7:
                command = "thumb_up"
                previousCoordx = 0
                previousCoordy = 0
                mouse.wheel(delta=2)

            #else:
                #command = "none"
        else:
            command = None
        ###FPS COUNTER WAS USED IN TESTING CHECK HOW THAT APPLICATION WAS RUNNING OPTIMALLY
        # FPS Counter
        #currentTime = time.time()
        #fps = 1 / (currentTime - previousTime)
        #previousTime = currentTime
        try:
            cv2.putText(flip_img, str(command), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        except:
            cv2.putText(flip_img, str("None"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        ###HOW TO EXIT THE GESTURE CAPTURING PROCESS
        if keyboard.is_pressed("q"):
            # Key was pressed
            cv2.destroyAllWindows()
            break

        cv2.imshow("Image", flip_img)
        cv2.waitKey(1)

###SPOTIFY PLAY NEXT
def playnext(x):
    x.post('https://api.spotify.com/v1/me/player/next')
###SPOTIFY PLAY PREVIOUS
def playback(x):
    x.post('https://api.spotify.com/v1/me/player/previous')
###SPOTIFY PAUSE
def playpause(x):
    x.put('https://api.spotify.com/v1/me/player/pause')
###SPOTIFY RESUME
def playresume(x):
    x.put('https://api.spotify.com/v1/me/player/play')
###SPOTIFY VOL DOWN
def playdown(x, vol):
    vol = vol - 10
    query = (f'https://api.spotify.com/v1/me/player/volume?volume_percent={vol}')
    x.put(query)
###SPOTIFY VOL UP
def playup(x, vol):
    vol = vol + 10
    query = (f'https://api.spotify.com/v1/me/player/volume?volume_percent={vol}')
    x.put(query)
###GET SPOTIFY INFO TO CALCULATE CURRENT VOLUME
def spotify_vol_status(x):
    request = x.get('https://api.spotify.com/v1/me/player')
    response = request.json()
    #print(response)
    volume = response['device']['volume_percent']
    return volume

###SPOTIFY CAPTURING SCRIPT
def spotify_capture():
    load_dotenv()



    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    device_name = os.getenv("device_name")
    redirect_uri = os.getenv("redirect_uri")
    scope = os.getenv("scope")
    username = os.getenv("username")
    token_url = os.getenv("token_url")
    authorization_base_url = os.getenv("authorization_base_url")
    ###GETTING CREDENTIALS FOR SPOTIFY ACCOUNT BY USER LOGGING VIA OAUTH
    spotify = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)

    # Redirect user to Spotify for authorization
    authorization_url, state = spotify.authorization_url(authorization_base_url)
    print('Please go here and authorize: ', authorization_url)

    # Get the authorization verifier code from the callback url
    redirect_response = input('\n\nPaste the full redirect URL here: ')

    from requests.auth import HTTPBasicAuth

    auth = HTTPBasicAuth(client_id, client_secret)

    # Fetch the access token
    token = spotify.fetch_token(token_url, auth=auth, authorization_response=redirect_response)

    #print(token)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)


    ###SETTING UP PARAMTERS WHICH RESTRICT HOW MANY API REQUESTS CAN BE MADE, TO HELP PERFORMANCE AND
    ###INCREASE THE FUCNTIONALITY OF THE APPLICATION
    global counterPause
    global counterBack
    global counterPlay
    global counterNext
    global counterUp
    global counterDown
    counterPause =20
    counterBack = 20
    counterPlay = 20
    counterNext = 20
    counterUp = 20
    counterDown = 20


    while True:
        canSend = True
        success, img = cap.read()
        flip_img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        arrayPoints = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                baseCoords = [0, 0]
                # print(baseCoords)
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    height, width, c = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if id == 8:
                        currentX = cx
                        currentY = cy

                    if id == 0:
                        baseCoords = [int(cx), int(cy)]
                        arrayPoints.append([0, 0])
                    else:
                        xdiff = baseCoords[0] - cx
                        ydiff = baseCoords[1] - cy
                        arrayPoints.append([xdiff, ydiff])

                mpDraw.draw_landmarks(flip_img, handLms, mpHands.HAND_CONNECTIONS)

            arrayPointsList = list(
                itertools.chain.from_iterable(arrayPoints))
            # print(arrayPointsList)

            max_value = max(list(map(abs, arrayPointsList)))

            def CALCULATING_DISTANCE_(n):
                return n / max_value

            arrayPointsList = list(map(CALCULATING_DISTANCE_, arrayPointsList))
            arrayPointsList = numpy.asarray(arrayPointsList)
            # print(nparr)

            try:
                arrayPointsList = arrayPointsList.reshape(1, 42)
                prediction = keras_model_spotify.predict(arrayPointsList, verbose=0)
                prediction = prediction.tolist()
                gesture = prediction[0].index(max(prediction[0]))
                #print(prediction)
            except:
                command = None



            ###SHOWS THE USER WHICH GESTURE IS BEING RECOGNISED AND
            ###PERFORMS THE CORRESPONDING ACTION
            #print(gesture)
            if gesture == 0:
                command = "Back"
                if counterBack > 0:
                    counterBack = counterBack -1
                else:
                    playback(spotify)
                    counterBack = 40

            elif gesture == 1:
                command = "Next"

                if counterNext > 0:
                    counterNext = counterNext -1
                else:
                    playnext(spotify)
                    counterNext = 40




            elif gesture == 2:
                command = "Pause"
                if counterPause > 0:
                    counterPause = counterPause -1
                else:
                    playpause(spotify)
                    counterPause = 40



            elif gesture == 3:
                command = "Play"
                if counterPlay > 0:
                    counterPlay = counterPlay -1
                else:
                    playresume(spotify)
                    counterPlay = 40

            elif gesture == 4:
                command = "Vol Down"
                if counterDown > 0:
                    counterDown = counterDown -1
                else:
                    vol = spotify_vol_status(spotify)
                    playdown(spotify, vol)
                    counterDown = 20

            elif gesture == 5:
                command = "Vol Up"
                if counterUp > 0:
                    counterUp = counterUp -1
                else:
                    vol = spotify_vol_status(spotify)
                    playup(spotify, vol)
                    counterUp = 20

            elif gesture == 5:
                command = "Neutral"




            #else:
                #command = "none"
        else:
            command = None

        # FPS Counter
        #counterNext = counterNext - 1
        try:
            cv2.putText(flip_img, str(command), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        except:
            cv2.putText(flip_img, str("None"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        if keyboard.is_pressed("q"):
            # Key was pressed
            cv2.destroyAllWindows()
            break

        cv2.imshow("Image", flip_img)
        cv2.waitKey(1)

###CREATING THE INITIAL GUI WITH THE THREE BUTTONS TO CONTROL THE SYSTEM WITH
root = ctk.CTk()
root.geometry("750x450")
root.title("GestureControl")

title_label = ctk.CTkLabel(root, text="LiveFeed", font=ctk.CTkFont(size=30, weight="bold"))
title_label.pack(padx=10, pady=(40, 20))


add_button = ctk.CTkButton(root, text="Start PC Gesture Capture", width=500, command=lambda: start_capturing())
add_button.pack(pady=20)

add_button = ctk.CTkButton(root, text="Start Spotify Gesture Control", width=500, command=lambda: spotify_capture())
add_button.pack(pady=20)

add_button = ctk.CTkButton(root, text="Open User Manual", width=500, command=lambda: open_win())
add_button.pack(pady=20)

root.mainloop()