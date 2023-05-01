import os
from torch import cuda

MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])
windows = (True if (os.name == 'nt') else False)
if windows:
    OS = 'windows'
else:
    OS = 'linux'

ls = os.sep
device = 'cuda' if cuda.is_available() else 'cpu'

#################################################################################################################################
# FOLDERS
#################################################################################################################################

MODEL_FOLDER_PATH = MAIN_PATH + "models" + ls
DATA_FOLDER_PATH = MAIN_PATH + "data" + ls
INPUT_FOLDER_PATH = MAIN_PATH + "input_files" + ls
OUTPUT_FOLDER_PATH = MAIN_PATH + "output_files" + ls

# Model sub folders
AUDIO_MODEL_FOLDER_PATH = MODEL_FOLDER_PATH + "audio" + ls
FACE_MODEL_FOLDER_PATH = MODEL_FOLDER_PATH + "face" + ls
AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH = MODEL_FOLDER_PATH + "audio_face_combined" + ls
TRANSCRIBE_MODEL_FOLDER_PATH = MODEL_FOLDER_PATH + "transcribe" + ls

# Data sub folders
PREPROCESSED_IMAGES_FOLDER_PATH = DATA_FOLDER_PATH + "preprocessed_images_data" + ls
PREPROCESSED_AUDIO_FOLDER_PATH = DATA_FOLDER_PATH + "preprocessed_audio_data" + ls
PREPROCESSED_AV_FOLDER_PATH = DATA_FOLDER_PATH + "preprocessed_AV_data" + ls

CLEANED_LABELLED_AUDIO_FOLDER_PATH = DATA_FOLDER_PATH + "cleaned_labelled_audio" + ls
ORIGINAL_AV_DATASET_PATH = DATA_FOLDER_PATH + "original_AV" + ls
__ORIGINAL_FACES_DATASET_PATH = DATA_FOLDER_PATH + "original_faces" + ls
EXTRACTED_AUDIO_FOLDER_PATH = DATA_FOLDER_PATH + "extracted_audio" + ls
__TRAINING_FACES_FOLDER_PATH = DATA_FOLDER_PATH + "training_faces" + ls
TRAINING_AV_FOLDER_PATH = DATA_FOLDER_PATH + "training_AV" + ls

# ________________________________________________________________________________________________________________________________
# Original datasets sub folders

# Audio/Video
RAVDESS_VIDEOS_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'RAVDESS' + ls
SAVEE_VIDEOS_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'SAVEE' + ls + 'AudioVisualClip' + ls
OMG_ORIGINAL_VIDEOS_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'OMG' + ls + 'videos' + ls
MELD_ORIGINAL_VIDEOS_FOLDER_PATH = [ORIGINAL_AV_DATASET_PATH + 'MELD' + ls + 'train_sent_emo' + ls,
                                    ORIGINAL_AV_DATASET_PATH + 'MELD' + ls + 'test_sent_emo' + ls]
CREMAD_VIDEOS_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'CREMA-D_video' + ls

SAVEE_ORIGINAL_AUDIO_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'SAVEE' + ls + 'AudioData' + ls
CREMAD_ORIGINAL_AUDIO_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'CREMA-D' + ls
TESS_ORIGINAL_AUDIO_FOLDER_PATH = ORIGINAL_AV_DATASET_PATH + 'TESS' + ls

# Faces
FER_FACES_FOLDER_PATH = __ORIGINAL_FACES_DATASET_PATH + 'FER' + ls
CK_FACES_FOLDER_PATH = __ORIGINAL_FACES_DATASET_PATH + 'CK+' + ls
RAF_FACES_FOLDER_PATH = __ORIGINAL_FACES_DATASET_PATH + 'RAF' + ls
# ________________________________________________________________________________________________________________________________
# Extracted folders

# Audio
RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH = EXTRACTED_AUDIO_FOLDER_PATH + "RAVDESS" + ls
SAVEE_EXTRACTED_AUDIO_FOLDER_PATH = EXTRACTED_AUDIO_FOLDER_PATH + "SAVEE" + ls
CREMAD_EXTRACTED_AUDIO_FOLDER_PATH = EXTRACTED_AUDIO_FOLDER_PATH + "CREMA-D" + ls
TESS_EXTRACTED_AUDIO_FOLDER_PATH = EXTRACTED_AUDIO_FOLDER_PATH + "TESS" + ls

# Faces
FER_EXTRACTED_FACES_FOLDER_PATH = __TRAINING_FACES_FOLDER_PATH + "FER" + ls
CK_EXTRACTED_FACES_FOLDER_PATH = __TRAINING_FACES_FOLDER_PATH + "CK+" + ls
RAF_EXTRACTED_FACES_FOLDER_PATH = __TRAINING_FACES_FOLDER_PATH + "RAF" + ls

# AV
RAVDESS_EXTRACTED_AV_FOLDER_PATH = TRAINING_AV_FOLDER_PATH + "RAVDESS" + ls
SAVEE_EXTRACTED_AV_FOLDER_PATH = TRAINING_AV_FOLDER_PATH + "SAVEE" + ls
OMG_EXTRACTED_AV_FOLDER_PATH = TRAINING_AV_FOLDER_PATH + "OMG" + ls
MELD_EXTRACTED_AV_FOLDER_PATH = TRAINING_AV_FOLDER_PATH + "MELD" + ls
CREMAD_EXTRACTED_AV_FOLDER_PATH = TRAINING_AV_FOLDER_PATH + "CREMA-D" + ls

# All extracted
# These are the folders that are used for training
ALL_EXTRACTED_AUDIO_FOLDERS = [RAVDESS_EXTRACTED_AUDIO_FOLDER_PATH, SAVEE_EXTRACTED_AUDIO_FOLDER_PATH, CREMAD_EXTRACTED_AUDIO_FOLDER_PATH, TESS_EXTRACTED_AUDIO_FOLDER_PATH]
ALL_EXTRACTED_FACES_FOLDERS = [FER_EXTRACTED_FACES_FOLDER_PATH, CK_EXTRACTED_FACES_FOLDER_PATH, RAF_EXTRACTED_FACES_FOLDER_PATH]
ALL_EXTRACTED_AV_FOLDERS = [
    # MELD_EXTRACTED_AV_FOLDER_PATH,  # Disabled due to bad labels
    # OMG_EXTRACTED_AV_FOLDER_PATH,
    CREMAD_EXTRACTED_AV_FOLDER_PATH,
    RAVDESS_EXTRACTED_AV_FOLDER_PATH,
    SAVEE_EXTRACTED_AV_FOLDER_PATH,
]

##################################################################################################################################
# FILE PATHS
##################################################################################################################################

# Model file paths
AUDIO_MODEL_SAVE_PATH = AUDIO_MODEL_FOLDER_PATH + "audio_model.pth"
FACE_MODEL_SAVE_PATH = FACE_MODEL_FOLDER_PATH + "face_model.pth"
AUDIO_FACE_COMBINED_MODEL_SAVE_PATH = AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH + "audio_face_combined_model.pth"

# Tuner file paths
AUDIO_TUNER_SAVE_PATH = os.path.join(AUDIO_MODEL_FOLDER_PATH, "tuner")
AUDIO_TUNER_CSV_SAVE_PATH = os.path.join(AUDIO_MODEL_FOLDER_PATH, "tuner_results.csv")
FACE_TUNER_SAVE_PATH = os.path.join(FACE_MODEL_FOLDER_PATH, "tuner")
FACE_TUNER_CSV_SAVE_PATH = os.path.join(FACE_MODEL_FOLDER_PATH, "tuner_results.csv")
AUDIO_FACE_COMBINED_TUNER_SAVE_PATH = os.path.join(AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH, "tuner")
AUDIO_FACE_COMBINED_TUNER_CSV_SAVE_PATH = os.path.join(AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH, "tuner_results.csv")

# Normalization file paths
FACE_NORM_SCALAR_SAVE_PATH = FACE_MODEL_FOLDER_PATH + "face_norm_scalar.pkl"
AUDIO_FACE_COMBINED_NORM_SCALAR_SAVE_PATH = AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH + "audio_face_combined_norm_scalar.pkl"

# Best hyperparameters file paths
AUDIO_BEST_HP_JSON_SAVE_PATH = os.path.join(AUDIO_MODEL_FOLDER_PATH, "audio_best_hyperparameters.json")
FACE_BEST_HP_JSON_SAVE_PATH = os.path.join(FACE_MODEL_FOLDER_PATH, "face_best_hyperparameters.json")
AUDIO_FACE_COMBINED_BEST_HP_JSON_SAVE_PATH = os.path.join(AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH, "combined_best_hyperparameters.json")

# Class weights path
FACE_CLASS_WEIGHTS_SAVE_PATH = FACE_MODEL_FOLDER_PATH + "face_class_weights.pkl"

# FER dataset file paths
__OLD_FER_PATH = DATA_FOLDER_PATH + "fer2013.csv"
__NEW_FER_PATH = DATA_FOLDER_PATH + "fer2013new.csv"
FER_PATH = __OLD_FER_PATH

# OMG labels file path
OMG_labels_file_paths = [
    ORIGINAL_AV_DATASET_PATH + 'OMG' + ls + "omg_TrainVideos.csv",
    ORIGINAL_AV_DATASET_PATH + 'OMG' + ls + "omg_ValidationVideos.csv",
    ORIGINAL_AV_DATASET_PATH + 'OMG' + ls + "omg_TestVideos_WithLabels.csv"
]

MELD_labels_file_paths = [ORIGINAL_AV_DATASET_PATH + 'MELD' + ls + "train_sent_emo.csv",
                          ORIGINAL_AV_DATASET_PATH + 'MELD' + ls + "test_sent_emo.csv"]

# RAF faces labels file path
RAF_faces_labels_file_path = RAF_FACES_FOLDER_PATH + ls + "list_partition_label.txt"

###################################################################################################################################

test_split_percentage = 0.15

MIN_VID_LEN = 1.5
VIDEO_ANALYSE_WINDOW_SECS = 4.0  # All videos and audio files in the training set should be of this length. If videos in the training set are longer, only the last x seconds will be used. (3.5 sec video, last 2.5 secs will be used)
FRAME_RATE = 4  # Frames per second
AUDIO_INPUT_SHAPE = (3, 126, 13)  # Find and change this shape if you change the window length
TEXT_SENTIMENT_INPUT_SHAPE = (9)

MAX_THREADS = 7

##################################################################################################################################

# Use the FULL_EMOTIONS_LIST for labelling the data when you format it for preprocessing. DO NOT USE THE 'SIMPLIFIED EMOTIONS_INDEX'. Simplifying is done automatically. You can change the simplified emotions list and it's mapping if you want.
FULL_EMOTION_INDEX = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
FULL_EMOTION_INDEX_REVERSE = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

SIMPLIFIED_EMOTIONS_MAP = {
    2: 0,  # Fear -> Sad/Fear
    4: 0,  # Sad -> Sad/Fear
    6: 1,  # Neutral -> Neutral
    3: 2,  # Happy -> Happy
    1: 4,  # Disgust -> Surprise/Disgust
    5: 4,  # Surprise -> Surprise/Disgust
    0: 3,  # Angry -> Angry
}

SIMPLIFIED_EMOTIONS_INDEX_REVERSE = {'Sad/Fear': 0, 'Neutral': 1, 'Happy': 2, 'Angry': 3, 'Surprise/Disgust': 4}
# Map full emotions to simplified emotions
for key, value in FULL_EMOTION_INDEX_REVERSE.items():
    SIMPLIFIED_EMOTIONS_INDEX_REVERSE[key] = SIMPLIFIED_EMOTIONS_MAP[value]

SIMPLIFIED_EMOTIONS_INDEX = {0: 'Sad/Fear', 1: 'Neutral', 2: 'Happy', 3: 'Angry', 4: 'Surprise/Disgust'}

# These labels will be considered unimportant and their weightage will be reduced to the average weightage of the other labels. Assuming these labels are low in frequency.
REDUCE_LABEL_WEIGHTAGE_TO_ONE = []

# Choose the emotion index to use
EMOTION_INDEX = SIMPLIFIED_EMOTIONS_INDEX
EMOTION_INDEX_REVERSE = SIMPLIFIED_EMOTIONS_INDEX_REVERSE

softmax_len = len(EMOTION_INDEX)
NON_SIMPLIFIED_SOFTMAX_LEN = len(FULL_EMOTION_INDEX)

# print("EMOTION_INDEX", EMOTION_INDEX)
# print("EMOTION_INDEX_REVERSE", EMOTION_INDEX_REVERSE)

# data filename label_schema:
# <dataset_name>_<s.no>_<emotion_index>.wav
