#!/opt/conda/envs/emo/bin/python

import numpy as np
import pandas as pd
import cv2
import dlib
import os
from tqdm import tqdm

SHAPE_PREDICTOR_LOCATION = "/home/util/shape_predictor_68_face_landmarks.dat"
DATA_CSV = './../../facial_expressions/data/legend.csv'
IMAGE_DIR_LOC = '/home/data/facial'

def split_labels():
    """
    Load in legend of data labels, shuffle, and return split data legend
    Returns:
        train/test/validation splits of data (image name, emotion label, emotion key)
    """
    # load data and drop irrelevant user.id column
    data = pd.read_csv(DATA_CSV).drop(labels='user.id', axis=1)

    # randomly shuffle all data
    data = data.sample(frac=1, random_state=0)

    # capitalize all emotion labels
    data['emotion'] = data['emotion'].str.upper()
    # Get names of indexes for which emotion is 'CONTEMPT'
    indexNames = data[data['emotion'] == 'CONTEMPT'].index
    # Delete these row indexes from dataFrame
    data.drop(indexNames , inplace=True)
    # factorize all keys to integers
    data['emotion_keys'], keys = pd.factorize(data['emotion'])

    # split into train/test/validation split 70/20/10
    length, _ = data.shape
    train, test, validation = np.split(data, [int(.7 * length), int(.9 * length)])

    return train, test, validation


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


def load_images(image_names, emotions, split):
    out_emotions = []
    idx = 0

    for image_name, emotion_key in tqdm(zip(image_names, emotions)):
        infile_location = './../../facial_expressions/images/'+image_name
        outfile_location = os.path.join(IMAGE_DIR_LOC, 'images', split, str(idx) + '_input.jpg')

        image = cv2.cvtColor(cv2.imread(infile_location), cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        try:
            masked = get_masked_face(gray_image)
            out_emotions.append(emotion_key)
            cv2.imwrite(outfile_location, masked)
            idx += 1
        except Exception as e:
            print(e)
            print('Cannot find face, skipping...')

    return out_emotions


def get_image_data(data_splits):
    """
    Note that emotion keys correspond to index in emotion array:
    ['HAPPINESS', 'NEUTRAL', 'SURPRISE', 'SADNESS',
        'ANGER', 'FEAR', 'DISGUST']

    Args:
        data_splits:

    Returns:
        List of data numpy arrays for loaded images modified to be masked
    """
    for split, data in data_splits.items():
        image_names = data['image'].tolist()
        emotions = data['emotion_keys'].tolist()
        out_emotions = load_images(image_names, emotions, split)
        np.save(os.path.join(IMAGE_DIR_LOC, 'images', split, 'labels.npy'), out_emotions)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(IMAGE_DIR_LOC, 'images')):
        os.makedirs(os.path.join(IMAGE_DIR_LOC, 'images', 'train'))
        os.makedirs(os.path.join(IMAGE_DIR_LOC, 'images', 'valid'))
        os.makedirs(os.path.join(IMAGE_DIR_LOC, 'images', 'test'))

    train, test, validation = split_labels()
    get_image_data({'train': train, 'valid': validation, 'test': test})
