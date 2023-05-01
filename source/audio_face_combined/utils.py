import csv
import os
import shutil
import threading
import time
import traceback

import cv2
import librosa
import matplotlib.pyplot as plt
import moviepy.editor as moviepy
import numpy as np

from moviepy.editor import VideoFileClip
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

# audio_analysis_utils imports
import source.audio_analysis_utils.audio_config as audio_config
import source.audio_analysis_utils.utils as audio_utils

# combination model imports
import source.audio_face_combined.combined_config as combined_config
import source.config as config

# face model imports
import source.face_emotion_utils.face_config as face_config


def create_folder(new_path):
    if not os.path.exists(new_path):
        print("Creating folder " + new_path)
        os.makedirs(new_path)


# ffmpeg: .ogv, .mp4, .mpeg, .avi, .mov etc. video to .mp4
def convert_video_to_mp4(video_path, save_path):
    clip = moviepy.VideoFileClip(video_path)
    clip.write_videofile(save_path, verbose=False, logger=None, threads=4)


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


def convert_video_to_x_fps(vidcap, fps_out, print_flag=True):
    fps_in = vidcap.get(cv2.CAP_PROP_FPS)
    if print_flag:
        print("fps_in: ", fps_in)
        print("fps_out: ", fps_out)

    index_in = -1
    index_out = -1

    frames = []
    while True:
        success = vidcap.grab()
        if not success: break
        index_in += 1

        out_due = int(index_in / fps_in * fps_out)
        if out_due > index_out:
            success, frame = vidcap.retrieve()
            if not success: break
            index_out += 1

            frames.append(frame)

    return frames


def __convert_video_to_frames__(cv2_capture, frame_rate, crop_video_length, return_consistent_length=False, print_flag=True):
    """
    If the video is shorter than frame_rate frames, it will be padded with images of zeros
    If the video is longer than frame_rate frames, it will be split into multiple sets of video_length.
    """

    frames = convert_video_to_x_fps(cv2_capture, frame_rate, print_flag=print_flag)

    frame_rate_crop_video_length = round(frame_rate * crop_video_length)  # Target length of each clip expressed in frames

    # split into multiple clips, with the smallest clip at the beginning
    frames_clips = []
    for i in range(len(frames), -1, -frame_rate_crop_video_length):
        if i - frame_rate_crop_video_length - 1 < 0:
            frames_clips.append(frames[0:i])
        else:
            frames_clips.append(frames[i - frame_rate_crop_video_length:i])
    frames_clips.reverse()

    if return_consistent_length:
        # make the smallest clip the same length
        for i in range(len(frames_clips[0]), frame_rate_crop_video_length):
            frames_clips[0].append(np.zeros(frames_clips[0][0].shape, dtype=type(frames_clips[0][0][0][0])))

    if print_flag:
        print("\n__convert_video_to_frames__")
        print("original clips len(frames): ", len(frames))
        print("len(frames_clips): ", len(frames_clips))
        len_of_clips = [len(frames_clips[i]) for i in range(len(frames_clips))]
        print("Small clip has length: ", min(len_of_clips), " at index: ", len_of_clips.index(min(len_of_clips)))
        print("Big clip has length: ", max(len_of_clips), " at index: ", len_of_clips.index(max(len_of_clips)))
        # print("Median clip has length: ", np.median(len_of_clips), " at index: ", len_of_clips.index(np.median(len_of_clips)))

    return frames_clips


def __convert_audio_to_clips__(signal, sample_rate, crop_audio_length, return_consistent_length=True, print_flag=True):
    """
    If the audio is shorter than sample_rate samples, it will be padded with zeros.
    If the audio is longer than sample_rate samples, it will be split into multiple sets of sample_rate *  samples.
    """
    sample_rate_crop_audio_length = round(sample_rate * crop_audio_length)  # Target length of each clip expressed in samples

    # split into multiple clips, with the smaller crop at the beginning
    signal_clips = []
    for i in range(len(signal), -1, -sample_rate_crop_audio_length):
        if i - sample_rate_crop_audio_length - 1 < 0:
            signal_clips.append(signal[0:i])
        else:
            signal_clips.append(signal[i - sample_rate_crop_audio_length:i])
    signal_clips.reverse()

    if return_consistent_length:
        # make the smallest clip the same length
        signal_clips[0] = audio_utils.make_signal_len_consistent(signal_clips[0], sample_rate_crop_audio_length)

    if print_flag:
        print("\n__convert_audio_to_clips__")
        print("original clips len(signal): ", len(signal))
        print("len(signal_clips): ", len(signal_clips))
        len_of_clips = [len(signal_clips[i]) for i in range(len(signal_clips))]
        print("Small clip has length: ", min(len_of_clips), " at index: ", len_of_clips.index(min(len_of_clips)))
        print("Big clip has length: ", max(len_of_clips), " at index: ", len_of_clips.index(max(len_of_clips)))
        # print("Median clip has length: ", np.median(len_of_clips), " at index: ", len_of_clips.index(np.median(len_of_clips)))

    return signal_clips


def convert_video_to_frames_and_audio(video_path, crop_video_length, frame_rate, sample_rate=audio_config.READ_SAMPLE_RATE, min_vid_len=config.MIN_VID_LEN, print_flag=True, recursion_depth=1):
    """
    returns list of sets of (frame_rate * video_length) frames and audio samples of (sample_rate * video_length).

    arrays_of_frames: list of numpy arrays of shape (frame_rate * crop_video_length, height, width)
         [(crop_video_length * frame_rate) frames, (crop_video_length * frame_rate) frames, ...]
            (num_crops, crop_video_length * frame_rate, height, width)
    arrays_of_audio: list of lists of samples list
         [[[(crop_video_length * sample_rate) samples], [(crop_video_length * sample_rate) samples], ...], ...]
            (num_crops, crop_video_length, sample_rate)
    """
    try:
        # extract frames from video and split into multiple clips
        cap = cv2.VideoCapture(video_path)
        # get length of video in secs
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
        # print("video_length: ", video_length)
        if video_length < min_vid_len:
            print("Video is too short, skipping")
            return None

        arrays_of_frames = __convert_video_to_frames__(cap, frame_rate, crop_video_length, print_flag=print_flag)

        tempfile = f"{str(time.time())[-10:]}.wav"

        # extract audio from video and split into multiple clips
        audio_utils.convert_video_to_audio(video_path, tempfile)
        audio, sample_rate = librosa.load(
            tempfile,
            sr=sample_rate
        )
        try:
            os.remove(tempfile)
        except:
            pass
        arrays_of_audio_clips = __convert_audio_to_clips__(audio, sample_rate, crop_video_length, print_flag=print_flag)

        if print_flag:
            print("\nconvert_video_to_frames_and_audio")
            print("len(arrays_of_frames): ", len(arrays_of_frames))
            print("len(arrays_of_audio_clips): ", len(arrays_of_audio_clips))

        return arrays_of_frames, arrays_of_audio_clips
    except:
        # recursively call this function until it works
        if recursion_depth > 4:
            traceback.print_exc()
            print("recursion_depth > 10")
            return None

        return convert_video_to_frames_and_audio(video_path, crop_video_length, frame_rate, sample_rate=sample_rate, print_flag=print_flag, recursion_depth=recursion_depth + 1)


def split_video_into_equal_parts(full_video="full.mp4", save_dir="", divide_into_secs=combined_config.VIDEO_ANALYSE_WINDOW_SECS, min_vid_len=config.MIN_VID_LEN):
    current_duration = VideoFileClip(full_video).duration
    print(f"current_duration: {current_duration}secs")
    divide_into_count = round(current_duration / divide_into_secs)
    single_duration = current_duration / max(divide_into_count, 1)
    current_video = f"{0}.mp4"

    # return if the videos is already split
    if os.path.exists(save_dir):
        print("save_dir already exists")
        durations = {}
        for file in os.listdir(save_dir):
            start = float(file.split(".")[0])
            durations[start] = [start, start + single_duration]
        file_names = [i for i in os.listdir(save_dir)]

        return durations, file_names

    delete_folder_contents(save_dir)
    create_folder(save_dir)

    durations = {}
    file_names = []

    if current_duration < divide_into_secs:
        # if the video is shorter than the desired length, just copy the video
        shutil.copy(full_video, save_dir + current_video)
        start = 0
        end = single_duration
        durations[round(start)] = [start, end]
        file_names.append(start)
        return durations, file_names

    while current_duration >= min_vid_len:
        start = max(current_duration - single_duration, 0.0)
        end = current_duration
        clip = VideoFileClip(full_video).subclip(start, end)
        current_duration -= single_duration
        current_video = f"{save_dir + str(round(start))}.mp4"
        file_names.append(current_video)
        tempfile = f"{str(time.time())[-10:]}.mp4"
        clip.to_videofile(current_video, codec="libx264", temp_audiofile=tempfile, remove_temp=True,
                          audio_codec='aac')

        durations[round(start)] = [start, end]

        print("-----------------###-----------------")

    return durations, file_names


def split_video_into_custom_parts(divide_array, save_names, full_video="full.mp4", save_dir="", delete_folder_contents_flag=False):
    """
    Parameters:
    divide_array: list of times to split the video. Seconds will be rounded to second decimal place. Extension must be included.
    save_names: list of names to save the video clips as. DO NOT INCLUDE EXTENSIONS SUCH AS .mp4. DO NOT INCLUDE FOLDER PATHS, this must be included in save_dir.

    Examples:
    split_video_into_custom_parts([[0.534646, 10.58999], [11.5454, 20.456464], [25, 30]],
                                  save_names=["0.53_10.58", "11.54_20.45", "25.0_30.0"],
                                  full_video=config.INPUT_FOLDER_PATH + "Obama_speech.mp4",
                                  save_dir=config.OUTPUT_FOLDER_PATH + "Obama_speech")
    Output:
    0,53_10,58.mp4, 11,54_20,45.mp4, 25,0_30,0.mp4
    """
    try:
        current_duration = VideoFileClip(full_video).duration
        print("current_duration: ", current_duration)

        if delete_folder_contents_flag:
            delete_folder_contents(save_dir)
            create_folder(save_dir)

        for i, (start_time, end_time) in enumerate(divide_array):
            start_time = round(start_time, 2)
            end_time = round(end_time, 2)

            save_name = save_names[i].replace(".", ",") + ".mp4"

            try:
                os.remove(save_name)
            except:
                pass

            clip = VideoFileClip(full_video).subclip(start_time, end_time)
            current_video = os.path.join(save_dir, save_name)
            tempfile = f"{str(time.time())[-10:]}.mp4"
            clip.to_videofile(current_video, codec="libx264", temp_audiofile=tempfile, remove_temp=True,
                              audio_codec='aac', verbose=False)

            # delete the audio file
            try:
                os.remove(tempfile)
            except:
                pass

            print("....")
    except:
        traceback.print_exc()


def get_class_weights(class_series, multi_class=True, one_hot_encoded=True):
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
    of appearance of the label when the dataset was processed.
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
        return dict(zip(class_labels, class_weights))


def get_input_shape(which_input):
    if which_input == 'landmarks_depths':
        return (face_config.LANDMARK_COMBINATIONS_DEPTHS_CNT)
    elif which_input == 'image':
        return (3, face_config.FACE_SIZE, face_config.FACE_SIZE)
    elif which_input == 'audio':
        return combined_config.AUDIO_INPUT_SHAPE
    elif which_input == 'text_sentiment':
        return combined_config.TEXT_SENTIMENT_INPUT_SHAPE


def visualise_emotions(video_path, csv_path):
    import vlc

    video_started = False
    def update_bar_chart(emotions, title="Emotion Percentage"):
        emotions_labels = list(emotions.keys())
        emotions_values = list(emotions.values())

        fig, ax = plt.subplots()
        rects = ax.bar(emotions_labels, emotions_values)
        ax.set_ylabel("Percentage")
        ax.set_title(title)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}%'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects)
        plt.show()

    def make_all_emotions_zero(emotions):
        emotions_cpy = emotions.copy()
        for emotion in emotions_cpy:
            emotions_cpy[emotion] = 0
        return emotions_cpy

    def play_video(video_path):
        nonlocal video_started

        # creating vlc media player object
        media_player = vlc.MediaPlayer()
        # media object
        media = vlc.Media(video_path)
        # setting media to the media player
        media_player.set_media(media)
        # setting video scale
        media_player.video_set_scale(0.6)
        # setting audio track
        media_player.audio_set_track(1)
        # setting volume
        media_player.audio_set_volume(80)
        # start playing video
        video_started = True
        media_player.play()
        # wait so the video can be played for 5 seconds
        # irrespective for length of video
        time.sleep(5)

    chart_display_start_time = time.time()
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        emotions = None
        last_end_seconds = 0
        for i, row in enumerate(reader):
            if len(row) <= 2:
                continue

            if emotions is None:
                emotions = [h.split(":")[0] for h in row[2:]]
                print("Emotions: {}".format(emotions))
                input("Press Enter to start visualization...")

                thread = threading.Thread(target=play_video, args=(video_path,))
                thread.start()

                print("Waiting for video to start...")
                while not video_started:
                    time.sleep(0.1)


            start, end, *values = row
            emotions_values = {emotions[i]: float(val.split(":")[1].replace("%", "")) for i, val in enumerate(values)}
            # Get start and end time
            start_time = row[0]
            end_time = row[1]

            # Get time difference
            minutes, seconds = start_time.split(":")
            start_seconds = (int(minutes) * 60) + int(seconds)
            minutes, seconds = end_time.split(":")
            end_seconds = (int(minutes) * 60) + int(seconds)
            time_diff = end_seconds - start_seconds

            # Wait for start time in first row if it is not 0
            if (i == 0) and start_seconds != 0:
                empty_emotions = make_all_emotions_zero(emotions_values)
                update_bar_chart(empty_emotions, title=f"0 to {start}")
                time.sleep(start_seconds)
            # Wait for time difference between videos if some seconds are skipped
            elif i != 0 and last_end_seconds != start_seconds:
                print("Waiting for {} seconds. Skipped seconds".format(start_seconds - last_end_seconds))
                empty_emotions = make_all_emotions_zero(emotions_values)
                update_bar_chart(empty_emotions, title=f"Skipped {last_end_seconds} to {start_seconds}")
                time.sleep(start_seconds - last_end_seconds)

            # Wait for time difference
            update_bar_chart(emotions_values, title=f"{start} to {end}")
            print("Waiting for {} seconds. Playing for {} to {}".format(time_diff, start_time, end_time))
            time.sleep(time_diff)

            last_end_seconds = end_seconds
