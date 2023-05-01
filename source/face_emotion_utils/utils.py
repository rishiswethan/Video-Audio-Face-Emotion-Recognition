
import cv2
import math
import numpy as np
import traceback
import pickle
import source.config as config
import os
import json
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer

import source.face_emotion_utils.face_config as face_config

# save pickle object
def save_object(file_name, object):
    f = open(file_name, 'wb')
    pickle.dump(object, f)
    f.close()


# load pickle object
def load_object(file_name):
    f = open(file_name, 'rb')
    object = pickle.load(f)
    return object


def save_dict_as_json(file_name, dict, over_write=False):
    if os.path.exists(file_name) and over_write:
        with open(file_name) as f:
            existing_dict = json.load(f)

        existing_dict.update(dict)

        with open(file_name, 'w') as f:
            json.dump(existing_dict, f)
    else:
        with open(file_name, 'w') as f:
            json.dump(dict, f)


def load_dict_from_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
        # print(d)

    return d


def create_folder(new_path):
    if not os.path.exists(new_path):
        print("Creating folder " + new_path)
        os.makedirs(new_path)


# Goal of this function is to return a number between 1 and 0. Default is division with max, but you can use minmax_scaler as well
def normalise_lists(lists, save_min_max=False, norm_range=(0, 1), use_minmax=False, print_flag=True):
    def normalise(list, X_min, X_max, use_minmax=use_minmax):
        X = np.array(list)

        if use_minmax:
            X_std = (X - X_min) / (X_max - X_min)
            X_norm = X_std * (max_norm_range - min_norm_range) + min_norm_range
        else:
            # Divide by max value
            X_norm = X / X_max

        return X_norm

    min_norm_range, max_norm_range = norm_range

    if save_min_max:
        X_min = np.min(lists)
        X_max = np.max(lists)

        save_object(config.FACE_NORM_SCALAR_SAVE_PATH, (X_min, X_max))
    else:
        try:
            (X_min, X_max) = load_object(config.FACE_NORM_SCALAR_SAVE_PATH)
        except:
            traceback.print_exc()
            raise Exception("Make sure NORM_SCALAR was saved correctly from train data during training")

    if print_flag:
        print('Min: %f, Max: %f' % (X_min, X_max), ("minmax=", use_minmax))

    normalised_lists = []
    for i in range(len(lists)):
        X_norm = normalise(lists[i], X_min, X_max, use_minmax=use_minmax)
        normalised_lists.append(X_norm)

    return normalised_lists


def inverse_normalise(list, use_minmax=False):
    X = np.array(list).squeeze()
    (X_min, X_max) = load_object(config.NORM_SCALAR_SAVE_PATH)

    if use_minmax:
        og_val = (X * (X_max - X_min)) + X_min
    else:
        og_val = X * X_max

    return og_val


def euclidean_distance(coord_1, coord_2):
    x1, y1 = coord_1
    x2, y2 = coord_2

    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

def compute_distances(landmarks):

    dists = []
    X = []
    Y = []
    for landmark in landmarks:
        X.append(landmark[0])
        Y.append(landmark[1])

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if i == j:
                continue
            x1 = X[i]
            x2 = X[j]
            y1 = Y[i]
            y2 = Y[j]
            dist = euclidean_distance((x1, y1), (x2, y2))
            dists.append(dist)

    return dists


def get_as_softmax(label, count):
    full_soft = np.zeros(count)
    full_soft[label] = 1

    return full_soft


def pixel_to_image(pixels, name, len=48):
    pixels = list(str(pixels).split(" "))
    img_ar = []
    row = []
    for i in range(pixels.__len__()):
        row.append(int(pixels[i]))
        if (((i + 1) % len) == 0):
            img_ar.append(row)
            row = []
    img = np.array(img_ar)
    print(img.shape)
    cv2.imwrite(name + ".png", img)
    return img_ar


def get_input_shape(which_input):
    if which_input == 'landmarks_depths':
        return (face_config.LANDMARK_COMBINATIONS_DEPTHS_CNT)
    elif which_input == 'image':
        return (3, face_config.FACE_SIZE, face_config.FACE_SIZE)


def shuffle_train_data(array, seed=0):
    """
    Single array with be shuffled at axis 0
    Example:
        shuffle(X, seed=0)
        shuffle(Y, seed=0)
    """
    array = shuffle(array, random_state=seed)

    return array


def get_class_weights(class_series, multi_class=True, one_hot_encoded=True, normalize=True):
    """
    Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
    Some examples of different formats of class_series and their outputs are:
      - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
      - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
      {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
      - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
      {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
      - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
      {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
    The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
    of appareance of the label when the dataset was processed.
    In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
    """
    Y = np.array(class_series)
    labels_dict = {}
    for i in range(Y.shape[1]):
        cnt = 0
        for j in range(Y.shape[0]):
            if Y[j][i] == 1:
                cnt += 1
        labels_dict[i] = cnt

    print("labels_dict", labels_dict)

    if multi_class:
        # If class is one hot encoded, transform to categorical labels to use compute_class_weight
        if one_hot_encoded:
            class_series = np.argmax(class_series, axis=1)

        # Compute class weights with sklearn method
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
        for red_label in config.REDUCE_LABEL_WEIGHTAGE_TO_ONE:
            class_weights[red_label] = 1.0

        if normalize:
            class_weights = class_weights / np.sum(class_weights)

        return dict(zip(class_labels, class_weights))
    else:
        # It is neccessary that the multi-label values are one-hot encoded
        mlb = None
        if not one_hot_encoded:
            mlb = MultiLabelBinarizer()
            class_series = mlb.fit_transform(class_series)

        n_samples = len(class_series)
        n_classes = len(class_series[0])

        # Count each class frequency
        class_count = [0] * n_classes
        for classes in class_series:
            for index in range(n_classes):
                if classes[index] != 0:
                    class_count[index] += 1

        # Compute class weights using balanced method
        class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
        class_labels = range(len(class_weights)) if mlb is None else mlb.classes_

        if normalize:
            class_weights = class_weights / np.sum(class_weights)

        return dict(zip(class_labels, class_weights))


def find_filename_match(known_filename, directory):
    files_list = os.listdir(directory)
    for file_name in files_list:
        if known_filename in file_name:
            return os.path.join(directory, file_name)

    print("NO MATCH FOUND FOR: ", known_filename, "IN", directory)
    return None
