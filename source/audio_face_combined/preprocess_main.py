import cv2
import numpy as np
import os
import time
import _thread
import traceback

import source.config as config

# audio_face_combined imports
import source.audio_face_combined.utils as combined_utils
import source.audio_face_combined.combined_config as combined_config

# face_analysis_utils imports
import source.face_emotion_utils.face_mesh as face_mesh
import source.face_emotion_utils.face_config as face_config
import source.face_emotion_utils.utils as face_utils

# audio_analysis_utils imports
import source.audio_analysis_utils.utils as audio_utils
import source.audio_analysis_utils.preprocess_data as audio_preprocess
import source.audio_analysis_utils.audio_config as audio_config
import source.audio_analysis_utils.transcribe_audio as transcribe_audio

all_annotations_X = []
tl_br_xy_list = []

MAX_THREADS = combined_config.MAX_PREPROCESS_THREADS
results_thread = {}
num_threads_running = 0
detected_cnt = 1

# Helper to preprocess images in a set of frames of a clip
def preprocess_images(
        images_list,
        target_frames=round(combined_config.FRAME_RATE * combined_config.VIDEO_ANALYSE_WINDOW_SECS),
        print_flag=True,
        show_img=False,
):
    global all_annotations_X, tl_br_xy_list

    all_face_land_dists_depths_X = []  # List of all the face landmarks distances and depths
    all_face_images_X = []  # List of all images
    annotations_X = []  # List of all annotations
    tl_br_xy_X = []  # List of all top left coordinates

    no_detect_imgs_cnt = 0
    for image in images_list:
        # Get face mesh and cropped face
        results = face_mesh.get_mesh(image.copy(), return_mesh=show_img, upscale_landmarks=False, print_flag=print_flag)
        if results is None:
            no_detect_imgs_cnt += 1
            if print_flag:
                print("No mesh detected " + str(no_detect_imgs_cnt))

            # If no face is detected, add a zero image and a zero landmarks array
            grey_image = np.zeros((face_config.FACE_SIZE, face_config.FACE_SIZE))
            land_dists = np.zeros(face_config.LANDMARK_COMBINATIONS_DEPTHS_CNT)
        else:
            if show_img:
                land_dists, image, annotated_image, (tl_xy, br_xy) = results
                annotations_X.append(annotated_image)
                tl_br_xy_X.append((tl_xy, br_xy))
            else:
                land_dists, image = results

            # Convert cropped image to grayscale
            if len(image.shape) > 2:
                grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                grey_image = image
            grey_image = cv2.resize(grey_image, (face_config.FACE_SIZE, face_config.FACE_SIZE))

        # Add to list
        all_face_land_dists_depths_X.append(land_dists)
        all_face_images_X.append(grey_image)

    if no_detect_imgs_cnt > (len(all_face_images_X) / 2):
        print(f"More than half of the frames in the clip have no face detected. Skipping")
        return None

    # If there are fewer frames than the target, add zero frames
    if len(all_face_land_dists_depths_X) < target_frames:
        if print_flag:
            print(f"{len(all_face_land_dists_depths_X)} frames detected. Adding {target_frames - len(all_face_land_dists_depths_X)} zero frames for target of {target_frames}")
        for _ in range(target_frames - len(all_face_land_dists_depths_X)):
            all_face_land_dists_depths_X.append(np.zeros(face_config.LANDMARK_COMBINATIONS_DEPTHS_CNT))
            all_face_images_X.append(np.zeros((face_config.FACE_SIZE, face_config.FACE_SIZE)))

    # Convert to numpy arrays
    all_face_land_dists_depths_X = np.array(all_face_land_dists_depths_X)
    all_face_images_X = np.array(all_face_images_X)

    if show_img:
        all_annotations_X.append(annotations_X)
        tl_br_xy_list.append(tl_br_xy_X)

    return all_face_land_dists_depths_X, all_face_images_X


# Helper to preprocess audio in a single sample of a clip
def preprocess_audio(
        audio_samples,
        N_FFT,
        NUM_MFCC,
        HOP_LENGTH,
        return_audio=False,
        print_flag=True,
):
    # Clean the audio clip
    samples, sample_rate = audio_preprocess.clean_single(file_or_samples=audio_samples,
                                                         sample_rate=audio_config.READ_SAMPLE_RATE,
                                                         print_flag=print_flag)
    print()
    # Extract the 2D mfccs. These will be treated as images
    extracted_mfcc = audio_utils.extract_mfcc(
        (samples, sample_rate),
        N_FFT=N_FFT,
        NUM_MFCC=NUM_MFCC,
        HOP_LENGTH=HOP_LENGTH,
        print_flag=print_flag,
    )
    if print_flag:
        print(str(extracted_mfcc.shape) + "\n" + str(extracted_mfcc))

    if extracted_mfcc is None:
        return None

    if extracted_mfcc.shape[1] < NUM_MFCC:
        input(f"Wrong shape of extracted_mfcc{extracted_mfcc.shape}\n This will cause problems. Press enter to continue")

    extracted_mfcc = np.array(extracted_mfcc)

    if return_audio:
        return extracted_mfcc, samples, sample_rate
    else:
        return extracted_mfcc


# Helper to get sentiment of a single audio clip's transcript. We use a combination of sentiment and emotion as inputs
def get_sentiment(
        audio,
        disable_sentiment=combined_config.DISABLE_SENTIMENT,
        print_flag=True
):
    if disable_sentiment:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Get text from audio and get sentiment
    text = transcribe_audio.transcribe_audio(audio=audio)
    sentiment = transcribe_audio.get_text_sentiment(text)
    print_sentiment = sentiment.copy()
    print_sentiment.pop("emotion_softmax")
    print_sentiment.pop("sentiment_softmax")
    if print_flag:
        print("transcribed text: " + text)
        print("sentiment: " + str(print_sentiment))

    return sentiment['emotion_sentiment_softmax']


def preprocess_single_video(
        video_path,
        N_FFT,
        NUM_MFCC,
        HOP_LENGTH,
        label=None,
        print_flag=True,
        show_img=False,
):
    global results_thread, num_threads_running, detected_cnt

    try:
        # Get the last clip or the only clip
        result = combined_utils.convert_video_to_frames_and_audio(
            video_path=video_path,
            crop_video_length=combined_config.VIDEO_ANALYSE_WINDOW_SECS,
            frame_rate=combined_config.FRAME_RATE,
            sample_rate=audio_config.READ_SAMPLE_RATE,
            print_flag=print_flag,
        )
        if result is None:
            num_threads_running -= 1
            return None
        list_of_n_frames, list_of_audio_clips = result

        list_of_n_frames = list_of_n_frames[-1]
        list_of_audio_clips = list_of_audio_clips[-1]

        # Get face mesh and cropped face for each frame
        result = preprocess_images(
            images_list=list_of_n_frames,
            print_flag=print_flag,
            show_img=show_img
        )

        if result is None:
            num_threads_running -= 1
            return None

        frames_lands_X, frames_faceimages_X = result

        # Extract mfccs for the audio clip
        result = preprocess_audio(
            audio_samples=list_of_audio_clips.copy(),
            N_FFT=N_FFT,
            NUM_MFCC=NUM_MFCC,
            HOP_LENGTH=HOP_LENGTH,
            return_audio=True,
            print_flag=False,
        )
        if result is None:
            print(f"No mfcc detected in {video_path.split(config.ls)[-1]}. Skipping")
            num_threads_running -= 1
            return None
        else:
            extracted_mfcc_X, samples, sample_rate = result

        # Get sentiment of the transcription of audio clip
        text_sentiment_X = get_sentiment(
            audio=list_of_audio_clips,
            print_flag=True
        )

        frames_lands_X = np.array(frames_lands_X)
        frames_faceimages_X = np.array(frames_faceimages_X)
        extracted_mfcc_X = np.array(extracted_mfcc_X)
        text_sentiment_X = np.array(text_sentiment_X)

        if label is not None:
            if print_flag:
                print(f"Label: {label}")
            results_thread[video_path.split(os.sep)[-1]] = (frames_lands_X, frames_faceimages_X, extracted_mfcc_X, text_sentiment_X, label)
        else:
            results_thread[video_path.split(os.sep)[-1]] = (frames_lands_X, frames_faceimages_X, extracted_mfcc_X, text_sentiment_X)

        num_threads_running -= 1
        detected_cnt += 1

        return frames_lands_X, frames_faceimages_X, extracted_mfcc_X, text_sentiment_X
    except:
        print(f"Error in preprocess_single_video")
        traceback.print_exc()
        num_threads_running -= 1
        return None


def preprocess_videos(
        N_FFT,
        NUM_MFCC,
        HOP_LENGTH,
        original_videos_folder=config.ALL_EXTRACTED_AV_FOLDERS,
        output_path=config.PREPROCESSED_AV_FOLDER_PATH,
        print_flag=True,
        predict_mode=False,
        show_img=False,
        destroy_transcribe_models=True
):
    global results_thread, num_threads_running, detected_cnt

    # Initialise the transcription and sentiment models
    if predict_mode:
        transcribe_audio.init(1)
    else:
        transcribe_audio.init(max_threads=MAX_THREADS)

    all_frames_face_lands_X = []  # List of all the face landmarks distances and depths in each video (videos_len, frames_len, face_landmarks_cnt)
    all_frames_face_images_X = []  # List of all images (videos_len, frames_len, face_size, face_size)
    all_extracted_mfcc_list = []  # List of all the extracted mfccs in each video (videos_len, mfccs_len, mfccs_cnt)
    all_text_sentiment_X = []  # List of all the text sentiment of each video (videos_len, emotions_len)
    all_video_emotions_Y = []  # List of all the emotions as softmax vectors (videos_len, emotions_len)
    all_train_test_classification_Y = []  # List of all the datapoints as train/test (videos_len)
    detected_cnt = 0

    # add train and test folders
    if not predict_mode:
        folders_temp = []
        for folder in original_videos_folder:
            folders_temp.append(folder + "train" + os.sep)
            folders_temp.append(folder + "test" + os.sep)
        original_videos_folder = folders_temp.copy()

    print(f"Preprocessing {len(original_videos_folder)} folders")
    all_cnt = 0
    for folder in original_videos_folder:
        print(folder)
        for file in os.listdir(folder):
            all_cnt += 1

    so_far_cnt = 0
    last_so_far_cnt = so_far_cnt
    total_frames_detected = 0
    valid_file_list = []
    invalid_file_list = []
    total_results = 0

    start_time = time.time()
    start_memory = audio_utils.get_memory_used_by_system()
    for folder in original_videos_folder:
        for file in os.listdir(folder):
            so_far_cnt += 1
            print(f"\nPreprocessing video {so_far_cnt}/{all_cnt}: {folder.split(config.ls)[-2]}/{file}")
            # Only the last clip is used as first clip is the smaller clip.
            # We are assuming that each clip in the training set is at worst, slightly longer than the window size

            try:
                num_threads_running += 1

                if predict_mode:
                    emotion_label_softmax = np.array([0] * len(face_config.EMOTION_INDEX_REVERSE))
                else:
                    # Get video emotion label
                    emotion_label = config.FULL_EMOTION_INDEX_REVERSE[file.split("_")[2].split(".")[0]]
                    emotion_label_softmax = face_utils.get_as_softmax(emotion_label, config.NON_SIMPLIFIED_SOFTMAX_LEN)
                    emotion_label_softmax = np.array(emotion_label_softmax)

                _thread.start_new_thread(preprocess_single_video, (
                    os.path.join(folder, file),
                    N_FFT,
                    NUM_MFCC,
                    HOP_LENGTH,
                    emotion_label_softmax,
                    False,
                    show_img,
                ))
            except:
                print(f"Error in {file}")
                invalid_file_list.append(file)
                num_threads_running -= 1
                continue

            printed_flag = False
            while num_threads_running >= MAX_THREADS:
                if print_flag:
                    if not printed_flag:
                        print(f"..........Waiting for threads to end. Num threads: {num_threads_running}")
                        printed_flag = True
                    if last_so_far_cnt != so_far_cnt:
                        last_so_far_cnt = so_far_cnt
                        print(f"{detected_cnt}/{so_far_cnt} videos detected")
                        stats_string = audio_utils.get_progress_stats(start_memory, start_time, so_far_cnt, all_cnt)
                        print(f"\n" + stats_string)

                time.sleep(0.1)

        wait_start_time = time.time()
        print_cnt = 0
        print_every = 2
        sleep_time = 0.1
        while num_threads_running > 0:
            if (time.time() - wait_start_time) > (MAX_THREADS * 10):
                num_threads_running = 0
                break
            if print_cnt % (print_every / sleep_time) == 0:
                print(f"Waiting for all {num_threads_running} threads to finish")

            print_cnt += 1
            time.sleep(sleep_time)

        for i, result_key in enumerate(results_thread):
            print(f"{folder.split(os.sep)[-3:-1]} Result: {i}, total: {total_frames_detected}: {result_key}")
            total_frames_detected += 1
            result = results_thread[result_key]

            if result is None:
                continue
            else:
                frames_face_lands_X, frames_face_images_X, extracted_mfcc_list, text_sentiment_X, emotion_label_softmax = result

            total_frames_detected += frames_face_lands_X.shape[0]
            valid_file_list.append(result_key) # result_key is the file name

            if print_flag:
                print(f"frames_face_lands_X.shape: {frames_face_lands_X.shape}")
                print(f"frames_face_images_X.shape: {frames_face_images_X.shape}")
                print(f"extracted_mfcc_list.shape: {extracted_mfcc_list.shape}")
                print(f"text_sentiment_X: {text_sentiment_X}")
                print(f"emotion_label_softmax: {emotion_label_softmax}")
                print(f"classification: {folder.split(config.ls)[-2]}")
                print(f"\ntotal_frames: {total_frames_detected}/{all_cnt * round(combined_config.VIDEO_ANALYSE_WINDOW_SECS * combined_config.FRAME_RATE)}")

                print("..........................................................")

            # Add to list
            all_frames_face_lands_X.append(frames_face_lands_X)
            all_frames_face_images_X.append(frames_face_images_X)
            all_extracted_mfcc_list.append(extracted_mfcc_list)
            all_text_sentiment_X.append(text_sentiment_X)
            all_video_emotions_Y.append(emotion_label_softmax)
            all_train_test_classification_Y.append(folder.split(config.ls)[-2])

        # Clear it for the next folder
        results_thread = {}
        result = None

    print(f"\nConverting {detected_cnt} videos to numpy arrays")
    all_frames_face_lands_X = np.array(all_frames_face_lands_X)
    all_frames_face_images_X = np.array(all_frames_face_images_X)
    all_extracted_mfcc_list = np.array(all_extracted_mfcc_list)
    all_text_sentiment_X = np.array(all_text_sentiment_X)
    all_video_emotions_Y = np.array(all_video_emotions_Y)
    print(f"Numpy arrays created")

    print("\nall_frames_face_lands_X shape: " + str(all_frames_face_lands_X.shape))
    print("all_frames_face_images_X shape: " + str(all_frames_face_images_X.shape))
    print("all_extracted_mfcc_list shape: " + str(all_extracted_mfcc_list.shape))
    print("all_text_sentiment_X shape: " + str(all_text_sentiment_X.shape))
    print("emotion_label_softmax shape: " + str(all_video_emotions_Y.shape))
    print("all_train_test_classification_Y shape: " + str(len(all_train_test_classification_Y)))

    if not predict_mode:
        print("\nSaving data to file")
        np.save(os.path.join(output_path, "all_frames_face_lands_X.npy"), all_frames_face_lands_X)
        np.save(os.path.join(output_path, "all_frames_face_images_X.npy"), all_frames_face_images_X)
        np.save(os.path.join(output_path, "all_extracted_mfcc_list.npy"), all_extracted_mfcc_list)
        np.save(os.path.join(output_path, "all_text_sentiment_X.npy"), all_text_sentiment_X)
        np.save(os.path.join(output_path, "all_video_emotions_Y.npy"), all_video_emotions_Y)
        np.save(os.path.join(output_path, "all_train_test_classification_Y.npy"), all_train_test_classification_Y)
        print("Preprocessing complete")

    if destroy_transcribe_models:
        print("Destroying transcribe models")
        transcribe_audio.delete_models()
        print("Transcribe models destroyed")

    return (all_frames_face_lands_X, all_frames_face_images_X, all_extracted_mfcc_list, all_text_sentiment_X, all_video_emotions_Y), valid_file_list, invalid_file_list


def load_preprocessed_data():
    all_frames_face_lands_X = np.load(os.path.join(config.PREPROCESSED_AV_FOLDER_PATH, "all_frames_face_lands_X.npy"))
    all_frames_face_images_X = np.load(os.path.join(config.PREPROCESSED_AV_FOLDER_PATH, "all_frames_face_images_X.npy"))
    all_extracted_mfcc_list = np.load(os.path.join(config.PREPROCESSED_AV_FOLDER_PATH, "all_extracted_mfcc_list.npy"))
    all_text_sentiment_X = np.load(os.path.join(config.PREPROCESSED_AV_FOLDER_PATH, "all_text_sentiment_X.npy"))
    all_video_emotions_Y = np.load(os.path.join(config.PREPROCESSED_AV_FOLDER_PATH, "all_video_emotions_Y.npy"))
    all_train_test_classification_Y = np.load(os.path.join(config.PREPROCESSED_AV_FOLDER_PATH, "all_train_test_classification_Y.npy"))

    # Simply the softmax from 7 classes to chosen number of classes
    all_video_emotions_Y = audio_utils.simply_emotion_softmax_list(all_video_emotions_Y)

    all_frames_face_lands_X = audio_utils.shuffle_train_data(all_frames_face_lands_X)
    all_frames_face_images_X = audio_utils.shuffle_train_data(all_frames_face_images_X)
    all_extracted_mfcc_list = audio_utils.shuffle_train_data(all_extracted_mfcc_list)
    all_text_sentiment_X = audio_utils.shuffle_train_data(all_text_sentiment_X)
    all_video_emotions_Y = audio_utils.shuffle_train_data(all_video_emotions_Y)
    all_train_test_classification_Y = audio_utils.shuffle_train_data(all_train_test_classification_Y)

    print("\nall_frames_face_lands_X shape: " + str(all_frames_face_lands_X.shape))
    print("all_frames_face_images_X shape: " + str(all_frames_face_images_X.shape))
    print("all_extracted_mfcc_list shape: " + str(all_extracted_mfcc_list.shape))
    print("all_text_sentiment_X shape: " + str(all_text_sentiment_X.shape))
    print("emotion_label_softmax shape: " + str(all_video_emotions_Y.shape))
    print("all_train_test_classification_Y shape: " + str(len(all_train_test_classification_Y)))

    return all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X, all_video_emotions_Y, all_train_test_classification_Y


def split_data(test_split_percent=audio_config.test_split_percentage):
    def classify(data, classification):
        classified_data_train = []
        classified_data_test = []
        for i, data_point in enumerate(data):
            if classification[i] == 'train':
                classified_data_train.append(data_point)
            else:
                classified_data_test.append(data_point)

        del data
        classified_data_train = np.array(classified_data_train)
        classified_data_test = np.array(classified_data_test)

        print("split data: " + str(classified_data_train.shape) + " " + str(classified_data_test.shape))

        return classified_data_train, classified_data_test

    # Load data from save
    all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X, all_video_emotions_Y, all_train_test_classification_Y = load_preprocessed_data()
    all_frames_face_images_X = all_frames_face_images_X / 255.0

    # Split data into train and test based of the all_train_test_classification_Y labels
    print("\nSplitting data")
    all_frames_face_images_X_train, all_frames_face_images_X_test = classify(all_frames_face_images_X, all_train_test_classification_Y)
    all_frames_face_lands_X_train, all_frames_face_lands_X_test = classify(all_frames_face_lands_X, all_train_test_classification_Y)
    all_extracted_mfcc_list_train, all_extracted_mfcc_list_test = classify(all_extracted_mfcc_list, all_train_test_classification_Y)
    all_text_sentiment_X_train, all_text_sentiment_X_test = classify(all_text_sentiment_X, all_train_test_classification_Y)
    all_video_emotions_Y_train, all_video_emotions_Y_test = classify(all_video_emotions_Y, all_train_test_classification_Y)

    # Make all single channel images 3 channel. Current shape is (n, 18, 64, 64) and we want (n, 18, 3, 64, 64)
    print("X_images shape before:", all_frames_face_images_X_train.shape)
    all_frames_face_images_X_train = np.repeat(all_frames_face_images_X_train[:, :, np.newaxis, :, :], 3, axis=2)
    print("X_images shape after:", all_frames_face_images_X_train.shape)

    print("X_images_test shape before:", all_frames_face_images_X_test.shape)
    all_frames_face_images_X_test = np.repeat(all_frames_face_images_X_test[:, :, np.newaxis, :, :], 3, axis=2)
    print("X_images_test shape after:", all_frames_face_images_X_test.shape)

    # Make them tuples for easier handling
    train_data = (all_frames_face_images_X_train, all_frames_face_lands_X_train, all_extracted_mfcc_list_train, all_text_sentiment_X_train, all_video_emotions_Y_train)
    test_data = (all_frames_face_images_X_test, all_frames_face_lands_X_test, all_extracted_mfcc_list_test, all_text_sentiment_X_test, all_video_emotions_Y_test)

    return train_data, test_data


# Only used for testing
if __name__ == '__main__':
    transcribe_audio.init(1)

    preprocess_single_video(
        video_path=config.INPUT_FOLDER_PATH + "SAVEE_182_Neutral.mp4",
        N_FFT=audio_config.tune_hp_ranges['N_FFT'][0],
        NUM_MFCC=audio_config.tune_hp_ranges['NUM_MFCC'][0],
        HOP_LENGTH=audio_config.tune_hp_ranges['HOP_LENGTH'][0],
        label=None,
        print_flag=True,
        show_img=True
    )