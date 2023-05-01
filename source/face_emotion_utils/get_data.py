import os
import warnings
import shutil
import csv

import source.config as config
import source.face_emotion_utils.face_config as face_config

def convert_newFER_data(folder_path, save_path):
    # <dataset_name>_<s.no>_[<emotion_index_1>, <emotion_index_2>,...<>].wav
    def get_label_schema(file_name, s_no, emotions):
        new_label = [0, 0, 0, 0, 0, 0, 0]
        for i, emotion in enumerate(emotions):
            label = EMOTION_INDEX_FER[i]
            label = face_config.EMOTION_INDEX_REVERSE[label]
            emo_intensity = emotion

            new_label[label] = emo_intensity

        label = f"FER_{file_name.split('.')[0]}_{new_label}.png"
        return label

    EMOTION_INDEX_FER = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Surprise',
        3: 'Sad',
        4: 'Angry',
        5: 'Disgust',
        6: 'Fear',
        7: 'Disgust',
    }  # mapping to fit to make it universal

    csv_list = list(csv.reader(open(config.__OLD_FER_PATH, 'r')))[1:]
    labels = {}
    for i, row in enumerate(csv_list):
        if row[1] == '':
            continue
        file_name = row[1][3:]
        file_num = int(file_name.split('.')[0])
        file_name = str(file_num) + '.png'
        labels[file_name] = [int(x) for x in row[2:9]]
        # print(file_name, labels[file_name])

    cnt = 0
    folder_files_list = os.listdir(folder_path)
    for i, file in enumerate(folder_files_list):
        if file not in labels:
            continue
        print("\nConverting " + file)
        print(file, labels[file])
        print(get_label_schema(file, cnt, labels[file]))
        shutil.copyfile(
            (folder_path + file),
            save_path + get_label_schema(file, cnt, labels[file])
        )
        cnt += 1

    print("Conversion complete")


def convert_oldFER_data(folder_path, save_path):

    EMOTION_INDEX_FER = {
        0:'Angry',
        1:'Disgust',
        2:'Fear',
        3:'Happy',
        4:'Sad',
        5:'Surprise',
        6:'Neutral'
    }

    def fix_nulls(s):
        for line in s:
            yield line.replace('\0', ' ')

    csv_list = list(csv.reader(fix_nulls(open(config.__OLD_FER_PATH, 'r'))))
    csv_file_emotions = {}
    for i, row in enumerate(csv_list):
        if i == 0 or row[0] == '':
            continue
        print(row[0])
        csv_file_emotions[i - 1] = EMOTION_INDEX_FER[int(row[0])]

    for i, file in enumerate(os.listdir(folder_path)):
        print(f"Converting file {i} " + file)
        file_name = int(file.split('.')[0])
        shutil.copyfile(
            (folder_path + file),
            save_path + f"FER_{i}_{csv_file_emotions[file_name]}.png"
        )


def convert_CK_data(folder_path, save_path):
    def get_label_schame(folder_name, s_no):
        label = EMOTION_INDEX_CK[folder_name]
        label = f"CK_{s_no}_{label}.png"
        return label

    EMOTION_INDEX_CK = {
        'anger': 'Angry',
        'contempt': 'Disgust',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'sadness': 'Sad',
        'surprise': 'Surprise',
    }

    cnt = 0
    for i, folder in enumerate(os.listdir(folder_path)):
        print("Converting " + folder)
        for j, file in enumerate(os.listdir(folder_path + folder + config.ls)):
            print("Converting " + file, save_path)
            shutil.copyfile(
                (folder_path + folder + config.ls + file),
                save_path + get_label_schame(folder, cnt)
            )
            cnt += 1


def convert_RAF_data(folder_path, save_path, label_file_path):
    def get_label_schame(file_label_dict, file_name, s_no):
        label = file_label_dict[file_name]
        emotion = EMOTION_INDEX_RAF[label]
        label = f"RAF_{s_no}_{emotion}.jpg"

        return label

    def get_file_label_dict(label_file_path):
        file_label_dict = {}
        with open(label_file_path, 'r') as f:
            for line in f:
                line = line.split(" ")
                file_label_dict[line[0]] = int(line[1])

        return file_label_dict

    EMOTION_INDEX_RAF = {
        6: 'Angry',
        3: 'Disgust',
        2: 'Fear',
        4: 'Happy',
        7: 'Neutral',
        5: 'Sad',
        1: 'Surprise',
    }

    file_label_dict = get_file_label_dict(label_file_path)

    cnt = 0
    for i, file in enumerate(os.listdir(folder_path)):
        print("Converting " + file)
        shutil.copyfile(
            (folder_path + file),
            save_path + get_label_schame(file_label_dict, file, cnt)
        )
        cnt += 1

# if __name__ == "__main__":
#     convert_RAF_data(folder_path=config.RAF_FACES_FOLDER_PATH + "images" + config.ls,
#                      save_path=config.RAF_EXTRACTED_FACES_FOLDER_PATH,
#                      label_file_path=config.RAF_faces_labels_file_path)
    # convert_newFER_data(config.__NEW_FER_PATH, config.__NEW_FER_SAVE_PATH)
    # convert_oldFER_data(config.__OLD_FER_PATH, config.__OLD_FER_SAVE_PATH)
    # convert_CK_data(config.__CK_PATH, config.__CK_SAVE_PATH)