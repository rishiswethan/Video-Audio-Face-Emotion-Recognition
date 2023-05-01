import moviepy.editor as mp
import numpy as np
import os
import json
import math
import random
import librosa
from sklearn.utils import shuffle
import moviepy.editor as mp
import time
import sys
import psutil
import shutil

import source.audio_analysis_utils.audio_config as audio_config
import source.config as config

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


def convert_video_to_audio(video_path, audio_path, sample_rate=audio_config.READ_SAMPLE_RATE):
    my_clip = mp.VideoFileClip(video_path)
    my_clip.audio.write_audiofile(audio_path, fps=sample_rate, verbose=False, logger=None)


def extract_audio_from_videos(videos_path, audios_path):
    videos = os.listdir(videos_path)

    for video in videos:
        print(video)
        convert_video_to_audio(videos_path + video, audios_path + video.split(".")[0] + '.wav')


def num_to_softmax(num, num_classes):
    softmax = np.zeros(num_classes)
    softmax[num] = 1
    return softmax


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


# recursively delete all content in a folder and the folder itself if chosen
def delete_folder_contents(folder_path, delete_folder=False):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

            if delete_folder:
                os.rmdir(folder_path)
    except Exception as e:
        pass


# Simply emotion label if the chosen emotion index is SIMPLIFIED_EMOTIONS_INDEX
def simply_emotion_softmax_list(all_emotions_Y):
    all_emotions_Y_temp = []
    if config.EMOTION_INDEX_REVERSE != config.EMOTION_INDEX:
        for i, y in enumerate(all_emotions_Y):
            # print("\noriginal emotion: " + str(y))
            y = list(y).index(max(y))  # get the index of the max value
            y = config.SIMPLIFIED_EMOTIONS_MAP[y]  # get the simplified emotion
            y = num_to_softmax(y, config.softmax_len)  # get the softmax
            all_emotions_Y_temp.append(y)  # set the new value
            # print("simplified emotion: " + str(y))

    all_emotions_Y = np.array(all_emotions_Y_temp)

    return all_emotions_Y


def get_softmax_probs_string(softmax, class_names_list):
    class_name_prob_pairs = {}
    for i, prob in enumerate(softmax):
        class_name_prob_pairs[class_names_list[i]] = prob

    softmax_probs = ""
    for i, class_name in enumerate(class_names_list):
        softmax_probs += f"{class_name}: {round(class_name_prob_pairs[class_name] * 100, 2)}%\n"

    return softmax_probs


def load_dict_from_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
        # print(d)

    return d


def make_signal_len_consistent(signal, signal_len):
    def pad_signal(signal, pad_size, pad_value, left_right="left"):
        signal_pad = np.array([pad_value for i in range(pad_size)])

        if left_right == "left":
            signal = np.concatenate((signal_pad, signal))
        else:
            signal = np.concatenate((signal, signal_pad))

        return signal

    if len(signal) < signal_len:
        pad_size = signal_len - len(signal)
        signal = pad_signal(signal, pad_size, np.mean(signal))
    elif len(signal) > signal_len:
        signal = signal[-signal_len:]

    return signal


# function to extract mean and variance of Mel-Frequency Cepstrum Components (MFCCs)
def extract_mfcc(file_or_samples_and_sr, N_FFT, NUM_MFCC, HOP_LENGTH, print_flag=False):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    """
    print_flag = True

    if type(file_or_samples_and_sr) == str:
        # load audio file and slice it to ensure length consistency among different files
        signal, sample_rate = librosa.load(file_or_samples_and_sr)
    else:
        signal, sample_rate = file_or_samples_and_sr

    if print_flag:
        print(f"Signal shape: {signal.shape}, sample rate: {sample_rate}")

    if signal.shape[0] < N_FFT:
        print("N_FFT length too low, signal length: " + str(signal.shape[0]), "N_FFT: " + str(N_FFT))
        return None

    # drop audio files with less than pre-decided number of samples
    if len(signal) >= audio_config.SIGNAL_SAMPLES_TO_CONSIDER:
        # ensure consistency of the length of the signal
        signal = make_signal_len_consistent(signal, audio_config.CONSISTENT_SIGNAL_LENGTH)
        if print_flag:
            print(signal.shape)
            print(signal)

        # extract MFCCs
        print(f"Extracting MFCCs with N_FFT: {N_FFT}, NUM_MFCC: {NUM_MFCC}, HOP_LENGTH: {HOP_LENGTH}...")
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        MFCCs = MFCCs.T

        # Image.fromarray(MFCCs).show()
        if print_flag:
            print(f"MFCCs shape: {MFCCs.shape}")
            # print("MFCCs: " + str(MFCCs))

        return MFCCs


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


def get_input_shape(target_hp, origin_data_folder=config.CLEANED_LABELLED_AUDIO_FOLDER_PATH):
    origin_data_folder = origin_data_folder + "test" + os.sep
    file_name = os.listdir(origin_data_folder)[0]
    print(f"file_name: {file_name}")

    while True:
        extracted_mfcc = extract_mfcc(
            origin_data_folder + file_name,
            N_FFT=target_hp['N_FFT'],
            NUM_MFCC=target_hp['NUM_MFCC'],
            HOP_LENGTH=target_hp['HOP_LENGTH']
        )
        if extracted_mfcc is not None:
            break

    print(f"extracted_mfcc: {extracted_mfcc.shape}")

    input_shape = (3, extracted_mfcc.shape[0], extracted_mfcc.shape[1])
    print(f"New input_shape: {input_shape}")

    return input_shape


def get_eta(start_time, current_iteration, total_iterations):
    elapsed_time = time.time() - start_time
    eta = elapsed_time * (total_iterations - current_iteration) / (current_iteration + 1)

    string = f"{int(eta / 60)}mins {round(eta % 60)}secs"
    return eta, string


def get_minute_second_string(seconds):
    return f"{int(seconds / 60)}mins {round(seconds % 60)}secs"


def get_memory_used_by_objects(object_array):
    def get_memory_used_by_object(obj):
        if isinstance(obj, dict):
            return sum(get_memory_used_by_object(v) for v in obj.values())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return sum(get_memory_used_by_object(i) for i in obj)
        memory_used = sys.getsizeof(obj)
        # convert to GB
        memory_used = memory_used / 1024 / 1024 / 1024

        return memory_used

    try:
        total_memory_used = 0
        for obj in object_array:
            total_memory_used += get_memory_used_by_object(obj)
    except:
        print("Error in get_memory_used_by_objects", sys.exc_info()[0])
        total_memory_used = 1

    return total_memory_used


def get_memory_used_by_system():
    # convert to GB
    return psutil.virtual_memory()[3] / 1024 / 1024 / 1024


def get_progress_stats(start_mem_use, start_time, so_far_cnt, total_cnt):
    estimated_memory_use_by_objs = ((get_memory_used_by_system() - start_mem_use) / so_far_cnt) * total_cnt

    string = f"ETA: {get_eta(start_time, so_far_cnt, total_cnt)[1]}\n" \
             f"Current memory: {round(get_memory_used_by_system(), 3)}GB; Start memory: {round(start_mem_use, 3)}GB; "\
             f"Est memory use by objects: {round(estimated_memory_use_by_objs, 3)}GB; Est memory at end: {round(estimated_memory_use_by_objs + start_mem_use, 3)}GB"

    return string