import source.config as config

import source.face_emotion_utils.face_mesh as face_mesh
import source.face_emotion_utils.utils as utils
import source.face_emotion_utils.face_config as face_config

import source.audio_analysis_utils.utils as audio_utils

# import cv2.cv2 as cv2
import cv2
import os
import numpy as np


__batch_indices = {}  # Returns the folder that gets called for given batch_num
__temp_loaded_data = None
__loaded_data_batch_folder = None

def load_preprocessed_data(
        normalise,
        input_path=config.PREPROCESSED_IMAGES_FOLDER_PATH,
        save_name_suffix="_default",
):
    # Load from file
    print(f"\nLoading preprocessed files from: {input_path}")
    save_folder = input_path

    X_landmark_depth = np.load(save_folder + f"X{save_name_suffix}.npy")
    X_images = np.load(save_folder + f"X_images{save_name_suffix}.npy")
    Y = np.load(save_folder + f"Y{save_name_suffix}.npy")
    print("Load complete")

    # Normalise distances
    if normalise:
        print("\nNormalising distances and images")
        X_landmark_depth = np.array(utils.normalise_lists(X_landmark_depth, save_min_max=True, use_minmax=False, print_flag=False))
        print("Normalisation complete")

    X_images = X_images / 255.0  # Normalise images by default

    return X_landmark_depth, X_images, Y


def save_preprocessed_data(
        X_landmark_depth,
        X_images,
        Y,
        save_name_suffix='_default',
        output_path=config.PREPROCESSED_IMAGES_FOLDER_PATH,
):

    print("\nConverting to numpy arrays")
    # Convert to numpy arrays
    X_landmark_depth = np.array(X_landmark_depth)
    X_images = np.array(X_images)
    Y = np.array(Y)
    print("Conversion complete")

    # Print to verify
    print(X_landmark_depth[:5])
    print(X_images[:5])
    print(Y[:5])
    print("\nShapes of arrays:")
    print(X_landmark_depth.shape)
    print(X_images.shape)
    print(Y.shape)

    # Save to file
    print(f"\nSaving preprocessed files to: {output_path}")
    save_folder = output_path
    utils.create_folder(new_path=save_folder)

    np.save(save_folder + f"X{save_name_suffix}.npy", X_landmark_depth)
    np.save(save_folder + f"X_images{save_name_suffix}.npy", X_images)
    np.save(save_folder + f"Y{save_name_suffix}.npy", Y)
    print("Save complete")


def preprocess_images(
        original_images_folders=config.ALL_EXTRACTED_FACES_FOLDERS,
        output_path=config.PREPROCESSED_IMAGES_FOLDER_PATH,
        print_flag=True,
):
    all_face_land_dists_depths_X = []  # List of all the face landmarks distances and depths
    all_face_images_X = []  # List of all images
    all_face_emotions_Y = []  # List of all the emotions as softmax

    all_cnt = 0
    for folder in original_images_folders:
        for file in os.listdir(folder):
            all_cnt += 1

    detected_cnt = 1
    so_far_cnt = 0
    for folder in original_images_folders:
        for file in os.listdir(folder):
            if print_flag:
                so_far_cnt += 1
                print(f"\nPreprocessing file {so_far_cnt}/{all_cnt}: {folder.split(config.ls)[-2]}/{file}")
                print(f"Detected cnt: {detected_cnt}/{so_far_cnt}")

            image = cv2.imread(folder + config.ls + file)

            # Get face mesh and cropped face
            results = face_mesh.get_mesh(image.copy(), showImg=False, upscale_landmarks=False)
            if results is None:
                if print_flag:
                    print("No mesh detected")
                    print(".....SKIPPING FILE", file)
                continue

            land_dists, image = results

            # Convert cropped image to grayscale
            if len(image.shape) > 2:
                grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                grey_image = image
            grey_image = cv2.resize(grey_image, (face_config.FACE_SIZE, face_config.FACE_SIZE))

            # Get face emotion label
            emotion_label = config.FULL_EMOTION_INDEX_REVERSE[file.split("_")[2].split(".")[0]]
            emotion_label_softmax = utils.get_as_softmax(emotion_label, config.NON_SIMPLIFIED_SOFTMAX_LEN)

            if print_flag:
                print(f"Image shape: {grey_image.shape}")
                print(f"Emotion label softmax: {emotion_label}, {emotion_label_softmax}")

            # Add to list
            all_face_land_dists_depths_X.append(land_dists)
            all_face_images_X.append(grey_image)
            all_face_emotions_Y.append(emotion_label_softmax)
            detected_cnt += 1


    print("Saving final data to file")
    # Save remaining data
    save_preprocessed_data(
        all_face_land_dists_depths_X,
        all_face_images_X,
        all_face_emotions_Y,
        output_path=output_path,
        save_name_suffix="_default",
    )
    print("Preprocessing complete")

    return all_face_land_dists_depths_X, all_face_images_X, all_face_emotions_Y


def split_data(X_lands, X_images, Y, test_split_percent=face_config.test_split_percentage):
    # Simply emotion labels
    Y = audio_utils.simply_emotion_softmax_list(Y)

    # Shuffle data with same seed
    X_lands = utils.shuffle_train_data(X_lands)
    X_images = utils.shuffle_train_data(X_images)
    Y = utils.shuffle_train_data(Y)

    # Make all single channel images 3 channel. Current shape is (n, 64, 64)
    print("X_images shape before:", X_images.shape)
    X_images = np.stack((X_images,) * 3, axis=-1)
    print("X_images shape after:", X_images.shape)

    # Split data into train and test
    train_len = int(len(X_lands) * (1 - test_split_percent))

    # Test data
    X_test_lands = X_lands[train_len:]
    X_test_images = X_images[train_len:]
    Y_test = Y[train_len:]

    # Train data
    X_lands = X_lands[:train_len]
    X_images = X_images[:train_len]
    Y = Y[:train_len]

    return (X_lands, X_images, Y), (X_test_lands, X_test_images, Y_test)