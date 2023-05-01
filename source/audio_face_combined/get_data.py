import os, shutil
import csv

import source.config as config
import source.audio_face_combined.utils as utils
import source.audio_face_combined.combined_config as combined_config

import source.audio_analysis_utils.utils as audio_utils

import threading
import time

file_and_save_names_pairs = {}
file_and_times_pairs = {}
cnt_rows = 0
last_file_name = ""


# Convert SAVEE data to acceptable format
def convert_SAVEE_data(folder_path, save_path):
    def get_label_schema(file_name, s_no):
        label = file_name.split('.')[0]

        emotion = label[0]
        if label[1].isalpha():
            emotion += label[1]

        emotion = EMOTION_INDEX_SAVEE[emotion]
        label = f"SAVEE_{s_no}_{emotion}.mp4"

        return label

    EMOTION_INDEX_SAVEE = {'a': 'Angry',
                           'd': 'Disgust',
                           'f': 'Fear',
                           'h': 'Happy',
                           'n': 'Neutral',
                           'sa': 'Sad',
                           'su': 'Surprise'}  # mapping to fit to make it universal

    # Delete existing folder's contents
    utils.delete_folder_contents(save_path, delete_folder=False)

    # Look through all sub-folders and convert them to acceptable format and save them in save_path
    cnt = 0
    savee_paths = os.listdir(folder_path)
    for i, folder in enumerate(savee_paths):
        sub_folder_path = os.listdir(folder_path + config.ls + folder + config.ls)
        for j, file in enumerate(sub_folder_path):
            # Convert the video to mp4
            print(f"Converting {folder + config.ls}" + file)
            utils.convert_video_to_mp4(video_path=folder_path + config.ls + folder + config.ls + file,
                                       save_path=save_path + get_label_schema(file, cnt))
            cnt += 1

    print("Conversion complete")


def convert_RAVDESS_data(folder_path, save_path):
    def get_label_schema(file_name, s_no):
        label_nums = file_name.split('.')[0].split("-")
        label_nums = [int(i) for i in label_nums]
        label = f"RAVDESS_{s_no}_{EMOTION_INDEX_REVDESS[label_nums[2]]}.mp4"

        return label

    EMOTION_INDEX_REVDESS = {1: 'Neutral',
                             2: 'Neutral',
                             3: 'Happy',
                             4: 'Sad',
                             5: 'Angry',
                             6: 'Fear',
                             7: 'Disgust',
                             8: 'Surprise'}  # mapping to fit to make it universal

    ravdess_paths = os.listdir(folder_path)
    for i, file in enumerate(ravdess_paths):
        print("Converting " + file)
        shutil.copyfile(folder_path + file, save_path + get_label_schema(file, i))
    print("Conversion complete")


def convert_OMG(folder_path, save_path, labels_csv, max_threads=10):
    threads_running = 0

    def split_video_into_custom_parts(
            divide_array,
            save_names,
            full_video,
            save_dir,
            delete_folder_contents_flag=True
    ):
        nonlocal threads_running

        threads_running += 1
        utils.split_video_into_custom_parts(
            divide_array=divide_array,
            save_names=save_names,
            full_video=full_video,
            save_dir=save_dir,
            delete_folder_contents_flag=delete_folder_contents_flag
        )
        threads_running -= 1


    # Get the file names and their corresponding emotions
    def get_file_time_emotion_pair(rows, max_time=combined_config.VIDEO_ANALYSE_WINDOW_SECS):
        global cnt_rows, file_and_save_names_pairs, file_and_times_pairs, last_file_name
        for i, row in enumerate(rows):
            if i == 0:
                continue

            file_name = row[0].split('=')[1] + ".mp4"
            emotion = int(row[7])
            emotion = EMOTION_INDEX_OMG[emotion]
            start_time = round(float(row[1]), 2)
            end_time = round(float(row[2]), 2)

            if last_file_name != file_name:
                file_and_save_names_pairs[file_name] = []
                file_and_times_pairs[file_name] = []
                last_file_name = file_name

            if end_time - start_time < max_time:
                file_name_save = f"OMG_{cnt_rows}-{start_time}-{end_time}_{emotion}"
                file_and_save_names_pairs[file_name].append(file_name_save)
                file_and_times_pairs[file_name].append((start_time, end_time))
                cnt_rows += 1
            else:
                print(f"\nmax_time:{max_time} is smaller than proposed time:{round(end_time - start_time, 2)}. Start time:{start_time}, End time:{end_time}, for {file_name}. Splitting the video into multiple videos")
                # Split the video into multiple videos if the time is greater than max_time
                start_time_loop = start_time
                end_time_loop = start_time_loop + max_time
                while True:
                    # Clip the video to max_time
                    if end_time_loop > end_time:
                        end_time_loop = end_time

                    # Remove videos that are too small
                    if end_time_loop - start_time_loop < (max_time / 2):
                        break

                    print(f"start_time:{start_time_loop}, end_time:{end_time_loop}")

                    file_name_save = f"OMG_{cnt_rows}-{start_time_loop}-{end_time_loop}_{emotion}"
                    file_and_save_names_pairs[file_name].append(file_name_save)

                    # If the end time is greater than the max time, then set the end time to max time
                    file_and_times_pairs[file_name].append((start_time_loop, end_time_loop))

                    # End the loop when end time is reached
                    if end_time_loop == end_time:
                        break

                    # Increment the start and end time
                    start_time_loop += max_time
                    end_time_loop += max_time
                    start_time_loop = round(start_time_loop, 2)
                    end_time_loop = round(end_time_loop, 2)

                    cnt_rows += 1

            # print(f"{i}) Converting {file_name} to {file_name_save}")
            # print(f"Start time: {start_time}, End time: {end_time}")
            # print()


    EMOTION_INDEX_OMG = {0: 'Angry',
                         1: 'Disgust',
                         2: 'Fear',
                         3: 'Happy',
                         4: 'Neutral',
                         5: 'Sad',
                         6: 'Surprise'}  # mapping to fit to make it universal

    # Get
    for csv_file in labels_csv:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            get_file_time_emotion_pair(rows)

    # Delete save folder's contents
    utils.delete_folder_contents(save_path, delete_folder=False)
    utils.create_folder(save_path)

    for i, file_name_target in enumerate(file_and_save_names_pairs):
        # Loop through array of save names
        print("-----------------------------------------------------------------------------------------------------------------------------------------")
        print(f"{i}/{len(file_and_save_names_pairs)} Converting {file_name_target}")
        # print conversions to verify
        for j, file_name_save in enumerate(file_and_save_names_pairs[file_name_target]):
            print(f"{i}_{j})Converting {file_name_target} to {file_name_save}")
            print(f"Start time: {file_and_times_pairs[file_name_target][j][0]}, End time: {file_and_times_pairs[file_name_target][j][1]}")
            print()

        # Check if target file exists
        if not os.path.exists(folder_path + file_name_target):
            print("Target file doesn't exist")
            continue

        while threads_running >= max_threads:
            time.sleep(0.1)

        threading.Thread(target=split_video_into_custom_parts,
                         args=((file_and_times_pairs[file_name_target],
                               file_and_save_names_pairs[file_name_target],
                               folder_path + file_name_target,
                               save_path,
                               False))).start()

    while threads_running > 0:
        print(f"Waiting for {threads_running} threads to finish")
        time.sleep(2)


def convert_MELD(folder_path, save_path, labels_csv, save_name):
    global cnt_rows

    # Get the file names and their corresponding emotions
    def get_file_savename_pair(rows):
        global file_and_save_names_pairs
        cnt_rows = 0
        for i, row in enumerate(rows):
            if i == 0:
                continue

            dia = row[5]
            utt = row[6]
            file_name = f"dia{dia}_utt{utt}.mp4"
            emotion = row[3]
            emotion = EMOTION_INDEX_MELD[emotion]

            save_n = f"MELD_{save_name}{cnt_rows}_{emotion}.mp4"
            file_and_save_names_pairs[file_name] = save_n

            cnt_rows += 1

        return file_and_save_names_pairs


    EMOTION_INDEX_MELD = {
        'disgust': 'Disgust',
        'fear': 'Fear',
        'joy': 'Happy',
        'neutral': 'Neutral',
        'sadness': 'Sad',
        'anger': 'Angry',
        'surprise': 'Surprise'
    }  # mapping to fit to make it universal

    with open(labels_csv, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        get_file_savename_pair(rows)

    for j, file_name_from in enumerate(file_and_save_names_pairs):
        file_name_to = file_and_save_names_pairs[file_name_from]
        print(f"{j})Converting {file_name_from} to {file_name_to}")
        print()
        if not os.path.exists(folder_path + file_name_from):
            continue

        shutil.copyfile(folder_path + file_name_from, save_path + file_name_to)


def convert_CREMAD(folder_path, save_path):
    global cnt_rows

    # Get the file names and their corresponding emotions
    def get_label_schema(file_name, cnt_rows):
        emotion = file_name.split('_')[-2]
        emotion = EMOTION_INDEX_CREMAD[emotion]

        save_n = f"CREMAD_{cnt_rows}_{emotion}.flv"
        return save_n

    EMOTION_INDEX_CREMAD = {
        'ANG': 'Angry',
        'DIS': 'Disgust',
        'FEA': 'Fear',
        'HAP': 'Happy',
        'NEU': 'Neutral',
        'SAD': 'Sad',
    }  # mapping to fit to make it universal

    cnt = 0
    folder_files_list = os.listdir(folder_path)
    for i, file in enumerate(folder_files_list):
        print("Converting " + file)
        shutil.copyfile(
            (folder_path + file),
            save_path + get_label_schema(file, cnt)
        )
        cnt += 1
    print("Conversion complete")


def split_to_test_and_train(data_path, test_path, train_path, test_per=config.test_split_percentage):
    data_paths = os.listdir(data_path)
    print("before shuffle")
    print(data_paths[:50])
    data_paths = audio_utils.shuffle_train_data(data_paths)
    print("after shuffle")
    print(data_paths[:50])

    for i, file in enumerate(data_paths):
        if os.path.isdir(data_path + config.ls + file):
            continue

        if i < len(data_paths) * test_per:
            print(f"Copying {file} to test")
            shutil.copyfile(
                (data_path + file),
                test_path + file
            )
        else:
            print(f"Copying {file} to train")
            shutil.copyfile(
                (data_path + file),
                train_path + file
            )
    print("Splitting complete")



if __name__ == "__main__":
    # convert_OMG(folder_path=config.OMG_ORIGINAL_VIDEOS_FOLDER_PATH, save_path=config.OMG_EXTRACTED_AV_FOLDER_PATH, labels_csv=config.OMG_labels_file_paths)
    # convert_RAVDESS_data(folder_path=config.RAVDESS_VIDEOS_FOLDER_PATH, save_path=config.RAVDESS_EXTRACTED_AV_FOLDER_PATH)
    # convert_SAVEE_data(folder_path=config.SAVEE_VIDEOS_FOLDER_PATH, save_path=config.SAVEE_EXTRACTED_AV_FOLDER_PATH)

    # convert_CREMAD(folder_path=config.CREMAD_VIDEOS_FOLDER_PATH,
    #                save_path=config.CREMAD_EXTRACTED_AV_FOLDER_PATH + os.sep + "all" + os.sep)

    # convert_MELD(folder_path=config.MELD_ORIGINAL_VIDEOS_FOLDER_PATH[0],
    #              save_path=config.MELD_EXTRACTED_AV_FOLDER_PATH + "all" + config.ls,
    #                 labels_csv=config.MELD_labels_file_paths[0],
    #                 save_name="tr")

    # convert_MELD(folder_path=config.MELD_ORIGINAL_VIDEOS_FOLDER_PATH[1],
    #              save_path=config.MELD_EXTRACTED_AV_FOLDER_PATH + "all" + config.ls,
    #                 labels_csv=config.MELD_labels_file_paths[1],
    #                 save_name="te")

    # split_to_test_and_train(data_path=config.CREMAD_EXTRACTED_AV_FOLDER_PATH + "all" + config.ls,
    #                         test_path=config.CREMAD_EXTRACTED_AV_FOLDER_PATH + "test" + config.ls,
    #                         train_path=config.CREMAD_EXTRACTED_AV_FOLDER_PATH + "train" + config.ls)

    # split_to_test_and_train(config.MELD_EXTRACTED_AV_FOLDER_PATH + "all" + os.sep,
    #                         config.MELD_EXTRACTED_AV_FOLDER_PATH + "test" + os.sep,
    #                         config.MELD_EXTRACTED_AV_FOLDER_PATH + "train" + os.sep)

    # split_to_test_and_train(config.RAVDESS_EXTRACTED_AV_FOLDER_PATH + "all" + os.sep,
    #                         config.RAVDESS_EXTRACTED_AV_FOLDER_PATH + "test" + os.sep,
    #                         config.RAVDESS_EXTRACTED_AV_FOLDER_PATH + "train" + os.sep)

    convert_OMG(folder_path=config.OMG_ORIGINAL_VIDEOS_FOLDER_PATH, save_path=config.OMG_EXTRACTED_AV_FOLDER_PATH + "train" + os.sep, labels_csv=config.OMG_labels_file_paths)
    # convert_OMG(folder_path=config.OMG_ORIGINAL_VIDEOS_FOLDER_PATH, save_path=config.OMG_EXTRACTED_AV_FOLDER_PATH + "test" + os.sep, labels_csv=config.OMG_labels_file_paths[2:])
    #
    # split_to_test_and_train(config.OMG_EXTRACTED_AV_FOLDER_PATH + "all" + os.sep,
    #                         config.OMG_EXTRACTED_AV_FOLDER_PATH + "test" + os.sep,
    #                         config.OMG_EXTRACTED_AV_FOLDER_PATH + "train" + os.sep)

# <dataset_name>_<s.no>_<emotion_index>.wav