import cv2
import numpy as np
import os
import csv
import librosa
import torch

import source.audio_face_combined.utils as combined_utils
import source.audio_face_combined.combined_config as combined_config
import source.audio_face_combined.preprocess_main as combined_preprocess_main
import source.audio_face_combined.model as model

import source.audio_analysis_utils.utils as audio_utils
import source.audio_analysis_utils.transcribe_audio as transcribe_audio

import source.face_emotion_utils.utils as face_utils

import source.config as config


def apply_bounding_box(image, prediction_softmax, tl_xy, br_xy):
    annotated_image = image.copy()

    prediction_index = list(prediction_softmax).index(max(prediction_softmax))
    pred_string = f"{config.EMOTION_INDEX[prediction_index]} : {round(max(prediction_softmax) * 100, 2)}%"

    face_input = annotated_image.copy()
    face_input = cv2.rectangle(face_input, tl_xy, br_xy, (0, 255, 0), max(face_input.shape[0] // 500, 1))
    cv2.putText(img=face_input,
                text=pred_string,
                org=(tl_xy[0], round(tl_xy[1] * 0.9)),
                fontFace=cv2.QT_FONT_NORMAL,
                fontScale=max(face_input.shape[0] // 1000, 1),
                color=(0, 255, 0),
                thickness=max(face_input.shape[0] // 500, 1))

    cv2.imshow("Video ", face_input)
    wait_time = round(round(1000 / combined_config.FRAME_RATE))
    cv2.waitKey(wait_time)


def predict_video(video_path, video_buffer_folder=config.MAIN_PATH + "VideoBufferFolder" + os.sep, model_save_path=config.AUDIO_FACE_COMBINED_MODEL_SAVE_PATH):
    # show_image = bool(int(input("\n\nShow annotated frames?\nEnter 1 or 0 for true or false: ")))
    show_image = False

    video_buffer_folder = video_buffer_folder + video_path.split(os.sep)[-1].split(".")[0] + os.sep
    video_name = video_path.split(os.sep)[-1]

    # Convert to videos of x secs
    durations, filename_list = combined_utils.split_video_into_equal_parts(video_path, video_buffer_folder)

    # Load tuned hyperparameters for audio and combined models to preprocess data
    # Audio
    audio_best_hyperparameters = face_utils.load_dict_from_json(config.AUDIO_BEST_HP_JSON_SAVE_PATH)
    print(f"Audio best hyperparameters, {audio_best_hyperparameters}")
    N_FFT = audio_best_hyperparameters["N_FFT"]
    HOP_LENGTH = audio_best_hyperparameters["HOP_LENGTH"]
    NUM_MFCC = audio_best_hyperparameters["NUM_MFCC"]

    # Combined
    best_hyperparameters = face_utils.load_dict_from_json(config.AUDIO_FACE_COMBINED_BEST_HP_JSON_SAVE_PATH)
    print(f"Combined best hyperparameters, {best_hyperparameters}")
    result, valid_file_list, invalid_file_list = combined_preprocess_main.preprocess_videos(
        N_FFT=N_FFT,
        NUM_MFCC=NUM_MFCC,
        HOP_LENGTH=HOP_LENGTH,
        original_videos_folder=[video_buffer_folder],
        print_flag=True,
        predict_mode=True,
        show_img=show_image,
    )

    combined_model = torch.load(model_save_path)
    combined_model.to(config.device).eval()

    if show_image == 1:
        annotated_frames = combined_preprocess_main.all_annotations_X
        tl_br_xy_list = combined_preprocess_main.tl_br_xy_list
        print(f"annotated_frames: {len(annotated_frames)}")
        print(f"tl_br_xy_list: {len(tl_br_xy_list)}")

    frames_face_lands_X, frames_face_images_X, extracted_mfcc_list, text_sentiment_X,  Y = result
    Y = np.array([0])  # Dummy Y

    frames_face_images_X = frames_face_images_X / 255.0

    print("frames_face_lands_X.shape: ", frames_face_lands_X.shape)
    print("frames_face_images_X.shape: ", frames_face_images_X.shape)
    print("extracted_mfcc_list.shape: ", extracted_mfcc_list.shape)
    print("text_sentiment_X.shape: ", text_sentiment_X.shape)
    print("Y.shape: ", Y.shape)

    # convert the images to 3 channels if they are not (1, num_frames, 3, square_size, square_size)
    if frames_face_images_X.shape[1] != 3:
        frames_face_images_X = frames_face_images_X[:, :, np.newaxis, :, :]
        frames_face_images_X = np.repeat(frames_face_images_X, 3, axis=2)
        print("New frames_face_images_X.shape: ", frames_face_images_X.shape)

    # convert to torch tensors
    frames_face_images_X = torch.from_numpy(frames_face_images_X).float().to(config.device)
    frames_face_lands_X = torch.from_numpy(frames_face_lands_X).float().to(config.device)
    extracted_mfcc_list = torch.from_numpy(extracted_mfcc_list).float().to(config.device)
    text_sentiment_X = torch.from_numpy(text_sentiment_X).float().to(config.device)

    # Predict
    predictions = combined_model(frames_face_images_X, frames_face_lands_X, extracted_mfcc_list, text_sentiment_X)
    predictions = torch.nn.functional.softmax(predictions, dim=1).detach().cpu().numpy()
    print(f"predictions: {predictions}")

    # Organize predictions and display them
    sum_probs = [0] * len(predictions[0])
    titles = list(config.EMOTION_INDEX.values())
    for i in range(len(predictions)):
        start = int(valid_file_list[i].split('.')[0])

        dur_i = start
        durations[dur_i][0] = f"{int(durations[dur_i][0] / 60)}:{int(durations[dur_i][0] % 60)}"
        durations[dur_i][1] = f"{int(durations[dur_i][1] / 60)}:{int(durations[dur_i][1] % 60)}"

        for j in range(len(predictions[i])):
            durations[dur_i].append(titles[j] + ": " + str(round(predictions[i][j] * 100, 1)) + "%")
            sum_probs[j] += predictions[i][j]

        predictions[i] = [round(p, 3) for p in predictions[i]]
        print("______________________________")
        print(f"predictions of file {valid_file_list[i]} from NN: ", predictions[i])
        print(f"Duration:\n{durations[dur_i][0]} to {durations[dur_i][1]}")
        print("\nPrediction probabilities:\n", audio_utils.get_softmax_probs_string(predictions[i], list(config.EMOTION_INDEX.values())))
        print("- File: ", valid_file_list[i])

        if show_image == 1:
            for j in range(len(annotated_frames[i])):
                apply_bounding_box(annotated_frames[i][j], predictions[i], tl_br_xy_list[i][j][0], tl_br_xy_list[i][j][0])
        print()

    # remove rows corresponding to invalid files
    duration_keys = list(durations.keys())
    duration_keys = sorted(duration_keys)
    temp_dict = {}
    for i in duration_keys:
        if len(durations[i]) > 2:
            temp_dict[i] = durations[i]
    durations = temp_dict

    # Write emotion prediction to csv file
    print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    with open(config.OUTPUT_FOLDER_PATH + f"video_output_{video_name.split('.')[0]}.csv", "w") as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        title = ["Start", "End"]
        print(title[0] + "\t" + title[1] + "\t\t", end="")
        print()
        writer.writerow(title)
        for duration in durations:
            print(str(durations[duration]).strip('[]').replace(", ", "\t\t"))
            writer.writerow(durations[duration])

    print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Overall video prediction probabilities:\n")
    for i in range(len(sum_probs)):
        print(f"{config.EMOTION_INDEX[i]} : {round((sum_probs[i] / len(predictions)) * 100, 2)}%")

    print("\n\nCheck the VideoBufferFolder for the cropped videos corresponding to the above file names.")

    print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Transcribe audio
    transcribe_audio.init()
    print("Transcribing audio...")
    audio_utils.convert_video_to_audio(video_path, config.OUTPUT_FOLDER_PATH + "audio.wav")
    audio, _ = librosa.load(config.OUTPUT_FOLDER_PATH + "audio.wav")
    video_transcript = transcribe_audio.transcribe_audio(audio=audio)
    try:
        os.remove(config.OUTPUT_FOLDER_PATH + "audio.wav")
    except:
        pass
    print("Transcript:\n")
    print(video_transcript)
    print("\n")

    # Write transcript to text file
    with open(config.OUTPUT_FOLDER_PATH + "video_transcript.txt", "w") as text_file:
        text_file.write(video_transcript)

    # Find common words in transcript
    print("Common words in transcript ranked:\n")
    common_words = transcribe_audio.find_common_words(video_transcript)
    print("Word list\t   Frequency")
    for word in common_words:
        print(word[0] + (" " * (11 - len(word[0]))), "\t", word[1])

    # Write word occurrences to csv file
    with open(config.OUTPUT_FOLDER_PATH + "transcript_word_occurrences.csv", "w") as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        title = ["Word", "Frequency"]
        writer.writerow(title)
        for word in common_words:
            writer.writerow(word)

    print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Check output folder for the output files, 'video_transcript.txt', 'transcript_word_occurrences.csv' and 'video_output.csv'")

    transcribe_audio.delete_models()
