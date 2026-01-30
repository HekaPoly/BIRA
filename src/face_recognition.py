# In order to get the key points we need to find the face in the image
# for that we can use opencv's built in har cascade or the pretrained model in dlib
# after that we need to use a facial landmarks detector for that we can also use dlib
# get more info here https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# api for dlib https://dlib.net/python/
# how to download dlib https://pyimagesearch.com/2017/03/27/how-to-install-dlib/ and https://github.com/davisking/dlib
# example https://dlib.net/face_landmark_detection.py.html
# the facial landmark model used can be find on https://dlib.net/files/ and is named
# shape_predictor_68_face_landmarks.dat.bz2 other model exits

# I aslo found another way to do it using MediaPipe Face Mesh
# the missing methods can be found here https://github.com/PyImageSearch/imutils

import numpy as np
import dlib
import cv2
import pyzed.sl as sl
from camera import Camera


if (__name__ == "__main__"):
    #load the pretrained face detector from dlib
    detector = detector = dlib.get_frontal_face_detector()
    #load the facial landmark predictor (97 358 KB)
    predictor = dlib.shape_predictor("/path-to-the-pretrained-model")

    # we need to aquire the image with the zed sdk
    myCamera = Camera()

    with myCamera:
        if (myCamera.grab() == sl.ERROR_CODE.SUCCESS):
            image = myCamera.get_frame()

    # we need resize the image
    resize(image, width = 500)
    #convert the image to grayscale
    grayScale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # upscale the image and get BB of the face can also work with RGB image
    face_BB = detector(grayScale_image, 1)

    for (index, BB) in face_BB:
        #get the facial landmarks coordinates (x,y)
        landmarks = predictor(grayScale_image, BB)
        #convert to np array
        landmarks = convertTonp(landmarks)
        #convert to openCv BB
        (x, y, w, h) = convertToOpenCVBB(BB)
        #draw the BB
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #draw the landmarks
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)    
    #show the result
    cv2.imshow("FaceID", image)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()



