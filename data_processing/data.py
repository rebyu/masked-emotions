import numpy as np
import pandas as pd
import cv2
import dlib
import os

SHAPE_PREDICTOR_LOCATION = "./../shape_predictor_68_face_landmarks.dat"
total_images = 0
img_idx = 0


def split_labels():
    """
    Load in legend of data labels, shuffle, and return split data legend
    Returns:
        train/test/validation splits of data (image name and emotion label)
    """
    # load data and drop irrelevant user.id column
    data = pd.read_csv('data/legend.csv').drop(labels='user.id', axis=1)

    # randomly shuffle all data
    data = data.sample(frac=1, random_state=0)

    # factorize all keys to integers
    data['emotion'] = data['emotion'].str.upper()
    data['emotion_keys'], keys = pd.factorize(data['emotion'])

    # split into train/test/validation split 70/20/10
    length, _ = data.shape
    global total_images
    total_images = length
    train, test, validation = np.split(data, [int(.7 * length), int(.9 * length)])

    return train, test, validation, keys


def get_masked_face(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_points = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_LOCATION)
    faces = detector(gray_image)

    for face in faces:
        landmarks = predictor(image=gray_image, box=face)
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
    masked_face = cv2.fillPoly(image, [mask_pts], (141, 197, 231))
    return masked_face


def load_images(image_names, emotion_keys):
    global img_idx
    images = []
    emotions = []
    for name, emotion in zip(image_names, emotion_keys):
        print(f'{img_idx + 1} of {total_images}')
        img_idx += 1
        file_location = f'images/{name}'
        image = cv2.cvtColor(cv2.imread(file_location), cv2.COLOR_BGR2RGB)
        try:
            masked = get_masked_face(image)
            images.append(masked)
            emotions.append(emotion)
        except Exception as e:
            print('Cannot find face, skipping...')
        images.append(masked)
        emotions.append(emotion)
    return images, emotions


def get_image_data(data_splits):
    """

    Args:
        data_splits:

    Returns:
        List of data numpy arrays for loaded images modified to be masked
    """
    data = []
    data_labels = []
    for d in data_splits:
        images = d[' pixels'].tolist()
        emotion_keys = d['emotion'].tolist()
        masked_images, emotions = load_images(image_names, emotion_keys)
        data.append(masked_images)
        data_labels.append(emotion)
    return data, data_labels


def print_keys(keys):
    idx = 0
    for k in keys:
        print(f'{idx}: \t{k}')
        idx += 1


def load_data():
    if os.path.exists('train_data.npy'):
        train_img = np.load('train_data.npy', allow_pickle=True)
        test_img = np.load('test_data.npy', allow_pickle=True)
        val_img = np.load('val_data.npy', allow_pickle=True)
        train_label = np.load('train_label.npy', allow_pickle=True)
        test_label = np.load('test_label.npy', allow_pickle=True)
        val_label = np.load('val_label.npy', allow_pickle=True)

        # keys = np.load('keys.npy')
        import ipdb; ipdb.set_trace()
    else:
        train, test, validation, keys = split_labels()
        data, data_labels = get_image_data([train, test, validation])
        (train_img, test_img, val_img) = data
        (train_label, test_label, val_label) = data_labels
        np.save('train_data.npy', train_img)
        np.save('test_data.npy', test_img)
        np.save('val_data.npy', val_img)
        np.save('train_label.npy', train_label)
        np.save('test_label.npy', test_label)
        np.save('val_label.npy', val_label)
        np.save('keys.npy', keys)
    print_keys(keys)

    return train_img, train_label,\
           test_img, test_label,\
           val_img, val_label


if __name__ == '__main__':
    load_data()
