import os
import dlib
import cv2

detector = dlib.get_frontal_face_detector()

for i in os.listdir('./imgs'):
    img = cv2.imread(f'./imgs/' + i)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dets = detector(gray_img,1)
    print("Number of faces detected:{}".format(len(dets)))
    
    for index,face in enumerate(dets):
        print( 'face {};left {};top {};right {};bottom {}'.format(index,face.left(),face.top(),face.right(),face.bottom()))
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),3)


    cv2.imshow('original img', img)
    cv2.imshow('gray img', gray_img)
    cv2.waitKey(0) 



