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
# emotion model link https://github.com/niebardzo/Emotions

import numpy as np
import dlib
import cv2
# import pyzed.sl as sl
# from camera import Camera
from joblib import load
from utils.image_processing import Face

def convert_dlib_landmarks_to_np_array(landmarks) :
    points = []
    for i in range(landmarks.num_parts):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    return np.array(points, dtype=np.int32)

def convert_dlib_BB_to_openCV_BB(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - rect.left()
    h = rect.bottom() - rect.top()
    return (x, y, w, h)

if (__name__ == "__main__"):
    #load the pretrained face detector from dlib
    detector = dlib.get_frontal_face_detector()
    #load the facial landmark predictor (97 358 KB)
    predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
    #load the emotion model
    emotion_model = load('../models/emotion.joblib')

    # we need to aquire the image with the zed sdk
    # myCamera = Camera()
    image = None

    # with myCamera:
    #     if (myCamera.grab() == sl.ERROR_CODE.SUCCESS):
    #         image = myCamera.get_frame()
    #     else:
    #         print("brooo the camera doesn't work :(")

    # resizing the image can positvely impact the computing time
    #convert the image to grayscale
    image = cv2.imread(r"C:\Users\Andy\Desktop\BIRA\src\Screenshot 2026-02-12 093020.png")
    grayScale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # upscale the image and get BB of the face can also work with RGB image
    face_BB = detector(grayScale_image, 1)
    if image is None:
        print("Image is None!")
    else:
        print("Image shape:", image.shape, "dtype:", image.dtype)
    print("number of faces detected: {}".format(len(face_BB)))
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    for BB in face_BB:
        face = Face(grayScale_image, BB, predictor)
        prediction = emotion_model.predict([face.extract_features()])
        #get the facial landmarks coordinates (x,y)
        #convert to openCv BB
        (x, y, w, h) = convert_dlib_BB_to_openCV_BB(BB)
        print(x, y, w, h)
        print("the emotion is {}".format(emotion_model.le.inverse_transform(prediction)[0]))
        #draw the BB
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
        cv2.putText(image, "###{}".format(emotion_model.le.inverse_transform(prediction)[0]), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #show the result
    cv2.imshow("Frame", image)
    
    while 1:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
#     /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator KNeighborsClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator GaussianNB from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator SVC from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator DecisionTreeClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator RandomForestClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator ExtraTreeClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator ExtraTreesClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator GradientBoostingClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator MLPClassifier from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(
# /home/nvidia/.local/lib/python3.10/site-packages/sklearn/base.py:442: InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.8.0 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
# https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
#   warnings.warn(