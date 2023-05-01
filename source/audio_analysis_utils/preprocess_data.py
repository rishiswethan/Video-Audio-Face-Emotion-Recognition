# import libraries
import numpy as np
import os
import librosa
import noisereduce as nr
from scipy.io import wavfile
import warnings
import time

import source.config as config
import source.audio_analysis_utils.audio_config as audio_config
import source.audio_analysis_utils.utils as utils

warnings.filterwarnings("ignore")


def clean_single(file_or_samples, save_path=None, print_flag=False, sample_rate=audio_config.READ_SAMPLE_RATE):
    if type(file_or_samples) == str:
        # load audio file at a sample rate of sample_rate Hz:
        y, sr = librosa.load(file_or_samples, sr=sample_rate)
        if print_flag:
            print(f"Cleaning {file_or_samples} of sr: {sr}")
    else:
        y = file_or_samples
        sr = sample_rate

    # Trim signal at a level of 20 db
    y_trim, _ = librosa.effects.trim(y, top_db=20)

    # Remove 25% noise from audio samples
    y_noise_rem = nr.reduce_noise(y=y_trim, sr=sr, prop_decrease=0.25, stationary=True, n_jobs=-1)

    if save_path is not None:
        wavfile.write(save_path, sr, y_noise_rem)

    return y_noise_rem, sr


# Function to clean the samples based on dataset and label it correctly
def clean_files(input_path, output_path, print_flag=False):  # Select files linked to a dataset
    files = os.listdir(input_path)  # List all files in the folder

    for i, file in enumerate(files):
        y_noise_rem, sr = clean_single(input_path + file, output_path + file, print_flag=print_flag)

        # Rename new file adding '_cleaned.wav' and put it in the new folder
        name = output_path + file

        #  Save output in a wav file in a new folder
        wavfile.write(name, sr, y_noise_rem)


# Takes a list of folders of audio files that are correctly labelled. Cleans them and returns a list of mfccs bundled in a numpy array and a list of labels
def preprocess_audio(
        NUM_MFCC,
        N_FFT,
        HOP_LENGTH,
        save_name_prefix,
        original_audio_folders=config.ALL_EXTRACTED_AUDIO_FOLDERS,
        cleaned_files_path=config.CLEANED_LABELLED_AUDIO_FOLDER_PATH,
        output_path=config.PREPROCESSED_AUDIO_FOLDER_PATH,
        force_clean_files=False,
        force_preprocess_files=False,
        print_flag=True,
):

    if os.path.exists(output_path + f"X_{save_name_prefix}.npy") and (not force_preprocess_files):
        print(f"\n\nFORCE_PREPROCESS_FILES IS FALSE\nPreprocessed files 'X_{save_name_prefix}.npy' already exist. Loading them...\n")
        X = np.load(output_path + f"X_{save_name_prefix}.npy")
        Y = np.load(output_path + f"Y_{save_name_prefix}.npy")
        Y_classifications = np.load(output_path + f"all_train_test_classification_Y_{save_name_prefix}.npy")

        print("X shape: " + str(X.shape))
        print("Labels shape: " + str(Y.shape))
        time.sleep(1)

        return X, Y, Y_classifications
    else:
        if force_preprocess_files:
            print(f"Forcing preprocessing of audio files... Files exist: {os.path.exists(output_path + f'X_{save_name_prefix}.npy')}")
        else:
            print(f"\n\nPreprocessed files 'X_{save_name_prefix}.npy' does not exist. Preprocessing them...")
        time.sleep(1)

    extracted_mfcc_list = []
    Y = []
    all_train_test_classification_Y = []  # List of all the datapoints as train/test (videos_len)

    # add train and test folders
    temp_folders = []
    for folder in original_audio_folders:
        temp_folders.append(folder + os.sep + "train" + os.sep)
        temp_folders.append(folder + os.sep + "test" + os.sep)
    original_audio_folders = temp_folders

    print(f"Preprocessing {len(original_audio_folders)} folders")
    all_cnt = 0
    for folder in original_audio_folders:
        print(folder)
        for file in os.listdir(folder):
            all_cnt += 1

    total_org_len = 0
    for folder in original_audio_folders:
        total_org_len += len(os.listdir(folder))

    # Clean the audio of all the dataset's folders if cleaned_files folder has fewer files than the original dataset
    total_clean_files_len = len(os.listdir(cleaned_files_path + "train")) + len(os.listdir(cleaned_files_path + "test"))
    if force_clean_files or (total_clean_files_len < total_org_len):
        for folder in original_audio_folders:
            if folder.split(os.sep)[-2] == "train":
                print("Cleaning files in " + folder)
                clean_files(input_path=folder,
                            output_path=cleaned_files_path + "train" + os.sep,
                            print_flag=print_flag)
            elif folder.split(os.sep)[-2] == "test":
                print("Cleaning files in " + folder)
                clean_files(input_path=folder,
                            output_path=cleaned_files_path + "test" + os.sep,
                            print_flag=print_flag)

    # Extract mfccs in train and test folders
    for train_test in ["train", "test"]:
        for j, file in enumerate(os.listdir(cleaned_files_path + train_test)):
            if print_flag:
                print(f"\nExtracting MFCCs from file num: {j} : " + cleaned_files_path + train_test + os.sep + file)
            extracted_mfcc = utils.extract_mfcc(
                cleaned_files_path + train_test + os.sep + file,
                N_FFT=N_FFT,
                NUM_MFCC=NUM_MFCC,
                HOP_LENGTH=HOP_LENGTH,
                print_flag=print_flag,
            )
            # print(str(extracted_mfcc.shape) + "\n" + str(extracted_mfcc))

            if extracted_mfcc is None:
                continue

            if extracted_mfcc.shape[1] < NUM_MFCC:
                input("Wrong shape")
                continue
            extracted_mfcc_list.append(extracted_mfcc)

            # Extract label from file name
            label = file.split("_")[2].split(".")[0]
            softmax = utils.num_to_softmax(num=config.FULL_EMOTION_INDEX_REVERSE[label],
                                           num_classes=config.NON_SIMPLIFIED_SOFTMAX_LEN)
            if print_flag:
                print(f"Label: {label}, {audio_config.EMOTION_INDEX_REVERSE[label]}\tSoftmax: {softmax}")
            Y.append(softmax)
            all_train_test_classification_Y.append(train_test)

    extracted_mfcc_list = np.array(extracted_mfcc_list)
    Y = np.array(Y)

    print(utils.get_class_weights(Y))

    print("Saving extracted MFCCs and labels to " + output_path)
    print("Extracted MFCCs shape: " + str(extracted_mfcc_list.shape))
    print("Labels shape: " + str(Y.shape))
    np.save(output_path + f"X_{save_name_prefix}.npy", extracted_mfcc_list)
    np.save(output_path + f"Y_{save_name_prefix}.npy", Y)
    np.save(output_path + f"all_train_test_classification_Y_{save_name_prefix}.npy", all_train_test_classification_Y)
    print("Saved extracted MFCCs and labels\n\n")

    return extracted_mfcc_list, Y, all_train_test_classification_Y


def split_data(X, Y, classifications, test_split_percent=audio_config.test_split_percentage):
    def classify(data, classification):
        classified_data_train = []
        classified_data_test = []
        for i, data_point in enumerate(data):
            if classification[i] == 'train':
                classified_data_train.append(data_point)
            else:
                classified_data_test.append(data_point)

        classified_data_train = np.array(classified_data_train)
        classified_data_test = np.array(classified_data_test)

        return classified_data_train, classified_data_test

    Y = utils.simply_emotion_softmax_list(Y)

    # Make all single channel images 3 channel. Current shape is (n, 64, 64)
    print("X_images shape before:", X.shape)
    X = np.stack((X,) * 3, axis=-1)
    print("X_images shape after:", X.shape)

    X_train, X_test = classify(X, classifications)
    Y_train, Y_test = classify(Y, classifications)

    return X_train, Y_train, X_test, Y_test
