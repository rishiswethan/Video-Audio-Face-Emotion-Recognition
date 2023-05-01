import os
import json
import source.config as config

def create_folder(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

print("Creating folders")
create_folder(config.INPUT_FOLDER_PATH)
create_folder(config.OUTPUT_FOLDER_PATH)
create_folder(config.MODEL_FOLDER_PATH)
create_folder(config.DATA_FOLDER_PATH)

create_folder(config.PREPROCESSED_IMAGES_FOLDER_PATH)
create_folder(config.PREPROCESSED_AV_FOLDER_PATH)
create_folder(config.PREPROCESSED_AUDIO_FOLDER_PATH)
create_folder(config.CLEANED_LABELLED_AUDIO_FOLDER_PATH)

create_folder(config.AUDIO_MODEL_FOLDER_PATH)
create_folder(config.FACE_MODEL_FOLDER_PATH)
create_folder(config.AUDIO_FACE_COMBINED_MODEL_FOLDER_PATH)
create_folder(config.MAIN_PATH + os.sep + "VideoBufferFolder")

print("Folders created")