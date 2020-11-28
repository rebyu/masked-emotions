import requests
from pathlib import Path

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np




SH_PRED_URL = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'
SH_PRED_FILE = './shape_predictor_68_face_landmarks.dat'

def get_detect_and_predict():
    if not Path(SH_PRED_FILE).exists():
        predicter_data = requests.get(SH_PRED_URL)

        with open(Path(SH_PRED_FILE), 'wb') as f:
            f.write(predicter_data.content)

    return dlib.get_frontal_face_detector(), dlib.shape_predictor(SH_PRED_FILE)


def get_face_points(img: np.ndarray):
    """
    Given an image represented in a numpy ndarray, returns a list of lists of tuples.
    Each sublist contains two-member tuples representing x, y positions of each facial point for a face,
    the indices corresponding to the appropiate facial point (see point_map.png).

    Params:
        img: A NumPy ndarray. Must be grayscaled.

    Returns:
        A list of lists of tuples.
    """

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_points = []

    detector, predictor = get_detect_and_predict()
    faces = detector(gray_img)

    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point

        landmarks = predictor(image=gray_img, box=face)
        points = []

        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            points.append((x, y))

        face_points.append(points)

    return face_points

if __name__ == "__main__":
    detector, predictor = get_detect_and_predict()
