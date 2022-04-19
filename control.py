import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import pynput
from tensorflow.keras.models import load_model
from pynput.keyboard import Key, Controller
import warnings


def run_controls():
      warnings.filterwarnings("ignore")

      #Initialize MediaPipe:
      mpHands = mp.solutions.hands
      hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
      mpDraw = mp.solutions.drawing_utils

      #Initialize Tensorflow:

      # Load the gesture recognizer model
      model = load_model('mp_hand_gesture')

      # Load class names
      f = open('gesture.names', 'r')
      classNames = f.read().split('\n')
      f.close()

      #Read frames from a webcam:
      # Initialize the webcam for Hand Gesture Recognition Python project
      #Press q to quit
      cap = cv2.VideoCapture(0)
      x_upper = 400
      x_lower = 290
      y_upper = 310
      y_lower = 225
      keyboard = Controller()
      key_action = 'run'

      while True:
         # Read each frame from the webcam
         _, frame = cap.read()
         y , x, c = frame.shape
         # Flip the frame vertically
         frame = cv2.flip(frame, 1)

         #Detect hand keypoints:
         framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         # Get hand landmark prediction
         result = hands.process(framergb)
         className = ''
         # post process the result
         landmarks = []
         if result.multi_hand_landmarks:
            #landmarks = []
            #print(len(result.multi_hand_landmarks))
            for handslms in result.multi_hand_landmarks:
               for lm in handslms.landmark:
                     #print(id, lm)
                     #h,w,_ = frame.shape
                     lmx = int(lm.x * x)
                     lmy = int(lm.y * y)
                     landmarks.append([lmx, lmy])
               # Drawing landmarks on frames
               #print(landmarks)
               mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
               
         #### Predict gesture in Hand Gesture Recognition project
            
         if landmarks!= []:   
            
            x1,y1 = landmarks[8][0],landmarks[8][1]  #index finger
            #creating circle at the tips of thumb and index finger
            cv2.circle(frame,(x1,y1),10,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
            #print(x1,y1)
            
            if x1 <= x_lower and y1 > y_lower and y1 < y_upper and key_action != "left":
               key_action = "left"
               keyboard.press(Key.left)
               keyboard.release(Key.left)
               print("left")
               
            elif x1 >= x_upper and y1 > y_lower and y1 < y_upper and key_action != "right":
               key_action = "right"
               keyboard.press(Key.right)
               keyboard.release(Key.right)
               print("right")
            
            elif y1 <=y_lower and x1 > x_lower and x1 < x_upper and key_action != "up":
               key_action = "up"
               keyboard.press(Key.up)
               keyboard.release(Key.up)
               print("up")
                     
            elif y1 >=y_upper and x1 > x_lower and x1 < x_upper and key_action != "down":
               key_action = "down" 
               keyboard.press(Key.down)
               keyboard.release(Key.down)
               print("down")
               
            #elif x1 > x_lower and x1 < x_upper and y1 > y_lower and y1 < y_upper:
            #   key_action = 'run'
            else:
               key_action = 'run'
               print("run")
               
            
            #prediction = model.predict([landmarks])
            #print(prediction)
            #classID = np.argmax(prediction)
            #className = classNames[classID]
            coords = str(x1)+' , '+ str(y1)
            ### show the prediction on the frame
            cv2.putText(frame, key_action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            #cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
               
         # Show the final output
         cv2.line(frame,(x_lower,10),(x_lower,470),(255,255,0,0.2),1)
         cv2.line(frame,(x_upper,10),(x_upper,470),(255,255,0,0.2),1)

         cv2.line(frame,(10,y_lower),(620,y_lower),(255,255,0,0.2),1)
         cv2.line(frame,(10,y_upper),(620,y_upper),(255,255,0,0.2),1)

         cv2.imshow("Output", frame)
               
         if cv2.waitKey(1) == ord('q'):
               break
         # release the webcam and destroy all active windows
      cap.release()
      cv2.destroyAllWindows()

