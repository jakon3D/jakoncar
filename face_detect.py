import os
import dlib
import cv2
import time

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
pTime = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    cTime = time.time()
    fps = 1 /(cTime - pTime)
    pTime = cTime
    cv2.putText(frame,'FPS: {}'.format(str(int(fps))),(5,15),font, .5, (255, 0, 255), 1) 
    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    dets = detector(gray_img,1)
    # print("Number of faces detected:{}".format(len(dets)))
    
    for index,face in enumerate(dets):
        # print( 'face {};left {};top {};right {};bottom {}'.format(index,face.left(),face.top(),face.right(),face.bottom()))
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(frame, (left,top),(right,bottom),(0,255,0),3)

    cv2.imshow('original img',frame) 

    if cv2.waitKey(1) & 0xff == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

