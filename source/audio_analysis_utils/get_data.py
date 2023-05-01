# importing libraries
import os
import warnings
import shutil

import source.config as config
import source.audio_analysis_utils.utils as utils

warnings.filterwarnings("ignore")


# label_schema -> <dataset_name>_<s.no>_<emotion_index>
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

# origin revdess indexing (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)


# Convert RAVDESS data to acceptable format
def convert_ravdess_vid2aud_data(folder_path, save_path):
    def get_label_schema(file_name, s_no):
        label_nums = file_name.split('.')[0].split("-")
        label_nums = [int(i) for i in label_nums]
        label = f"RAVDESS_{s_no}_{EMOTION_INDEX_REVDESS[label_nums[2]]}.wav"

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
        utils.convert_video_to_audio(folder_path + file, save_path + get_label_schema(file, i))
    print("Conversion complete")

# Convert SAVEE data to acceptable format
def convert_SAVEE_data(folder_path, save_path):
    def get_label_schema(file_name, s_no, folder_name):
        label = file_name.split('.')[0]
        emotion = label[:-2]
        emotion = EMOTION_INDEX_SAVEE[emotion]
        label = f"SAVEE-{folder_name}_{s_no}_{emotion}.wav"

        return label

    EMOTION_INDEX_SAVEE = {'a': 'Angry',
                           'd': 'Disgust',
                           'f': 'Fear',
                           'h': 'Happy',
                           'n': 'Neutral',
                           'sa': 'Sad',
                           'su': 'Surprise'}  # mapping to fit to make it universal

    cnt = 0
    savee_paths = os.listdir(folder_path)
    for i, folder in enumerate(savee_paths):
        sub_folder_path = os.listdir(folder_path + config.ls + folder + config.ls)
        for j, file in enumerate(sub_folder_path):
            print(f"Copying {folder + config.ls}" + file)
            shutil.copyfile(
                (folder_path + folder + config.ls + file),
                save_path + get_label_schema(file, cnt, folder)
            )
            cnt += 1
    print("Conversion complete")

# convert CREATE-DB data to acceptable format
def convert_CREMAD_data(folder_path, save_path):
    def get_label_schema(file_name, s_no):
        label = file_name.split('_')[2]
        emotion = label
        emotion = EMOTION_INDEX_CREMA[emotion]
        label = f"CREMA_{s_no}_{emotion}.wav"

        return label

    EMOTION_INDEX_CREMA = {
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


def convert_TESS_data(folder_path, save_path):
    def get_label_schema(file_name, s_no):
        label = file_name.split('.')[0].split('_')[2]
        emotion = label
        emotion = EMOTION_INDEX_TESS[emotion]
        label = f"TESS_{s_no}_{emotion}.wav"

        return label

    EMOTION_INDEX_TESS = {
        'angry': 'Angry',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        'ps': 'Neutral',
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
    data_paths = utils.shuffle_train_data(data_paths)
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


def convert_all_videos_to_audio(original_folder_path, save_folder_path):
    utils.delete_folder_contents(save_folder_path)

    file_paths = os.listdir(original_folder_path)
    for i, file in enumerate(file_paths):
        print("Converting", original_folder_path.split(config.ls)[-1] + config.ls + file,
              "to audio", save_folder_path.split(config.ls)[-1] + config.ls + file)
        utils.convert_video_to_audio(original_folder_path + file, save_folder_path + file.split('.')[0] + ".wav")

    print("Conversion complete")


# convert_all_videos_to_audio(original_folder_path=config.RAVDESS_EXTRACTED_AV_FOLDER_PATH + "train" + os.sep,
#                             save_folder_path=config.RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH + "train" + os.sep)
#
# convert_all_videos_to_audio(original_folder_path=config.RAVDESS_EXTRACTED_AV_FOLDER_PATH + "test" + os.sep,
#                             save_folder_path=config.RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH + "test" + os.sep)
#
#
#
# convert_all_videos_to_audio(original_folder_path=config.SAVEE_EXTRACTED_AV_FOLDER_PATH + "train" + os.sep,
#                             save_folder_path=config.SAVEE_EXTRACTED_AUDIO_FOLDER_PATH + "train" + os.sep)
#
# convert_all_videos_to_audio(original_folder_path=config.SAVEE_EXTRACTED_AV_FOLDER_PATH + "test" + os.sep,
#                             save_folder_path=config.SAVEE_EXTRACTED_AUDIO_FOLDER_PATH + "test" + os.sep)












# split_to_test_and_train(config.RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH,
#                         train_path=config.RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH + "train" + os.sep,
#                         test_path=config.RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH + "test" + os.sep)

# split_to_test_and_train(config.SAVEE_EXTRACTED_AUDIO_FOLDER_PATH + "all" + os.sep,
#                         train_path=config.SAVEE_EXTRACTED_AUDIO_FOLDER_PATH + "train" + os.sep,
#                         test_path=config.SAVEE_EXTRACTED_AUDIO_FOLDER_PATH + "test" + os.sep)

# split_to_test_and_train(config.CREMAD_EXTRACTED_AUDIO_FOLDER_PATH + "all" + os.sep,
#                         train_path=config.CREMAD_EXTRACTED_AUDIO_FOLDER_PATH + "train" + os.sep,
#                         test_path=config.CREMAD_EXTRACTED_AUDIO_FOLDER_PATH + "test" + os.sep)

# split_to_test_and_train(config.TESS_EXTRACTED_AUDIO_FOLDER_PATH + "all" + os.sep,
#                         train_path=config.TESS_EXTRACTED_AUDIO_FOLDER_PATH + "train" + os.sep,
#                         test_path=config.TESS_EXTRACTED_AUDIO_FOLDER_PATH + "test" + os.sep)

# convert_SAVEE_data(folder_path=config.SAVEE_ORIGINAL_AUDIO_FOLDER_PATH,
#                      save_path=config.SAVEE_EXTRACTED_AUDIO_FOLDER_PATH)

# convert_CREMAD_data(folder_path=config.CREMAD_ORIGINAL_AUDIO_FOLDER_PATH,
#                     save_path=config.CREMAD_EXTRACTED_AUDIO_FOLDER_PATH)

# convert_TESS_data(folder_path=config.TESS_ORIGINAL_AUDIO_FOLDER_PATH,
#                     save_path=config.TESS_EXTRACTED_AUDIO_FOLDER_PATH)