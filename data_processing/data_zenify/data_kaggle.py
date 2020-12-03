#!/opt/conda/bin/python

import numpy as np
import pandas as pd
import cv2
import dlib
import os
import sys
import subprocess
from tqdm import tqdm

SHAPE_PREDICTOR_LOCATION = "/home/util/shape_predictor_68_face_landmarks.dat"
DATA_CSV = '/home/data/icml_face_data.csv'   # https://www.kaggle.com/debanga/facial-expression-recognition-challenge
IMAGE_DIR_LOC = '/home/data'


def split_labels():
    """
    Load in legend of data labels, shuffle, and return split data legend
    Returns:
        train/test/validation splits of data (image name and emotion label)
    """
    # load data
    data = pd.read_csv(DATA_CSV)

    # randomly shuffle all data
    data = data.sample(frac=1, random_state=0)

    # factorize all keys to integers
    train_split = data[data['Usage'] == 'Training'].drop(labels='Usage', axis=1)
    valid_split = data[data['Usage'] == 'PublicTest'].drop(labels='Usage', axis=1)
    test_split = data[data['Usage'] == 'PrivateTest'].drop(labels='Usage', axis=1)

    return train_split, valid_split, test_split # <- dont know the keys thing


def get_masked_face(img):
    face_points = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_LOCATION)
    faces = detector(img)

    for face in faces:
        landmarks = predictor(image=img, box=face)
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
        face_points.append(points)

    # Get mask indices
    mask_idxs = list([x + 2 for x in range(13)])
    mask_idxs.append(28)

    # Get mask coordinates
    np_points = np.array(face_points)[0]
    mask_pts = np.array([np_points[x] for x in mask_idxs]).astype(np.int32)
    mask_pts = mask_pts.reshape((-1, 1, 2))

    # Draw filled polygon on image
    masked_face = cv2.fillPoly(img, [mask_pts], 0.6*141 + 0.3*197 + 0.1*231)
    return masked_face

def convert_to_image(txt):
    out = np.array([int(x) for x in txt.split(' ')]).reshape(48, 48)
    out = cv2.resize(np.float32(out), dsize=(110, 110), interpolation=cv2.INTER_CUBIC)
    return np.uint8(out)

def load_images(images, emotions, split):
    out_emotions = []
    idx = 0

    for image_txt, emotion in tqdm(zip(images, emotions)):
        file_location = os.path.join(IMAGE_DIR_LOC, 'images', split, str(idx) + '_input.jpg')
        image = convert_to_image(image_txt)

        try:
            masked = get_masked_face(image)
            out_emotions.append(emotion)
            cv2.imwrite(file_location, masked)
            idx += 1
        except Exception as e:
            print('Cannot find face, skipping...')

    return out_emotions


def get_image_data(data_splits):
    """

    Args:
        data_splits:

    Returns:
        List of data numpy arrays for loaded images modified to be masked
    """

    for split, data in data_splits.items():
        images = data['pixels'].tolist()
        emotions = data['emotion'].tolist()
        out_emotions = load_images(images, emotions, split)
        np.save(os.path.join(IMAGE_DIR_LOC, 'images', split, 'labels.npy'), out_emotions)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(IMAGE_DIR_LOC, 'images')):
        os.makedirs(os.path.join(IMAGE_DIR_LOC, 'images', 'train'))
        os.makedirs(os.path.join(IMAGE_DIR_LOC, 'images', 'valid'))
        os.makedirs(os.path.join(IMAGE_DIR_LOC, 'images', 'test'))

    train, validation, test = split_labels()
    get_image_data({'train': train, 'valid': validation, 'test': test})
