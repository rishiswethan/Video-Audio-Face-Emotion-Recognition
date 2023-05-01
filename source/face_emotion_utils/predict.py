import source.face_emotion_utils.utils as face_utils
import source.face_emotion_utils.face_mesh as face_mesh
import source.face_emotion_utils.face_config as face_config

import source.audio_analysis_utils.utils as audio_utils

import source.pytorch_utils.visualize as pt_vis

import source.config as config

import cv2
import numpy as np
from PIL import Image as ImagePIL
import time
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

FACE_SQUARE_SIZE = 64


def _create_gradcam(model, model_input, target_layer, device, verbose=False):
    return pt_vis.create_gradcam(model, model_input, target_layer, device, FACE_SQUARE_SIZE, verbose=verbose)


def _overlay_gradcam_on_image(img, grad_cam_pil, alpha=0.5, square_size=FACE_SQUARE_SIZE):
    return pt_vis.overlay_gradcam_on_image(img, grad_cam_pil, alpha=alpha, square_size=square_size)


def _visualise_feature_maps(feature_map, feature_map_name):
    pt_vis.visualise_feature_maps(feature_map, feature_map_name)


def _get_prediction(
        best_hp,
        img,
        model,
        imshow=False,
        video_mode=False,
        grad_cam=True,
        grad_cam_on_video=False,
        feature_maps_flag=True,
        device=config.device,
        verbose=True,
        emotion_index_dict=config.EMOTION_INDEX,
):
    try:
        # We detect the face and get the landmarks, regardless of if landmarks are used or not. This is because we need the face image for the model input
        result = face_mesh.get_mesh(image=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), upscale_landmarks=True, showImg=False, print_flag=True, return_mesh=True)
    except:
        raise Exception("Face mesh failed")
    if result is None:
        if verbose:
            print("No face detected")
        return None

    landmarks_depths, face_input_org, annotated_image, (tl_xy, br_xy) = result

    # Normalise if needed
    normalise = best_hp['normalise']
    if normalise:
        landmarks_depths = face_utils.normalise_lists([landmarks_depths], save_min_max=True, print_flag=verbose)[0]

    landmarks_depths = np.array(landmarks_depths)

    # Get the full image
    face_input = cv2.cvtColor(face_input_org, cv2.COLOR_BGR2GRAY)
    face_input = cv2.resize(face_input, (face_config.FACE_SIZE, face_config.FACE_SIZE))

    # Prep it for pytorch
    face_input = np.repeat(face_input[np.newaxis, :, :], 3, axis=0)
    if verbose:
        print("face_input.shape", face_input.shape)
    x = np.array(face_input)
    x = x / 255.
    x = x.reshape(face_utils.get_input_shape("image"))
    x = np.array(x[np.newaxis, :])
    if verbose:
        print(x.shape)

    landmarks_depths = np.array(landmarks_depths[np.newaxis, :])
    if verbose:
        print(landmarks_depths.shape)

    model_input = (x, landmarks_depths)

    # Get the prediction from the model
    pred = model(torch.from_numpy(np.array(model_input[0])).float().to(device),
                 torch.from_numpy(np.array(model_input[1])).float().to(device))
    pred = torch.nn.functional.softmax(pred, dim=1)
    if verbose:
        print("NN output:\n", pred)

    # Organise the prediction
    prediction_index = int(list(pred[0]).index(max(pred[0])))
    pred_numpy = pred[0].detach().cpu().numpy()

    if verbose:
        print("\nPrediction index: ", prediction_index)
        print("Prediction label: ", emotion_index_dict[prediction_index])
        print("Prediction probability: ", max(pred_numpy))
        print("\n\nPrediction probabilities:\n", audio_utils.get_softmax_probs_string(pred_numpy, list(emotion_index_dict.values())))

    string = audio_utils.get_softmax_probs_string(pred_numpy, list(emotion_index_dict.values()))
    string_img = emotion_index_dict[prediction_index] + ": " + str(round(max(pred_numpy) * 100)) + "%"

    return_objs = (emotion_index_dict[prediction_index], prediction_index, list(pred_numpy), img)

    if grad_cam:
        target_layer = model.base_model_conv.layer3

        grad_cam = _create_gradcam(model, model_input, target_layer, config.device)

        face_img = model_input[0][0]
        face_img = np.transpose(face_img, (1, 2, 0))
        face_img = face_img * 255.

        if verbose:
            print("face", face_img.shape)

        result_pil = _overlay_gradcam_on_image(face_img, grad_cam, alpha=0.5)

        if imshow and not video_mode:
            result_pil.show()

        result_npy = np.array(result_pil, dtype=np.uint8)

        if feature_maps_flag:
            # 
            target_layers = [
                model.base_model_conv.layer1,
                model.base_model_conv.layer2,
                model.base_model_conv.layer3,
                model.base_model_conv.layer4,
            ]
            for i, layer in enumerate(target_layers):
                if verbose:
                    print("Extracting feature maps from layer", i)
                feature_maps = []

                def hook_fn(module, input, output):
                    feature_maps.append(output.detach())

                layer.register_forward_hook(hook_fn)
                output = model(torch.from_numpy(np.array(model_input[0])).float().to(device),
                               torch.from_numpy(np.array(model_input[1])).float().to(device))

                _visualise_feature_maps(feature_maps[0], config.OUTPUT_FOLDER_PATH + "feature_maps_" + str(i) + ".png")
                layer._forward_hooks.clear()

        return_objs = (emotion_index_dict[prediction_index], prediction_index, list(pred_numpy), img, result_npy)

        if grad_cam_on_video:
            face_input = result_npy.copy()
        else:
            face_input = face_input_org.copy()

        face_input = cv2.rectangle(face_input,
                                   (face_input.shape[0] // 20, face_input.shape[0] // 20),
                                   (int(face_input.shape[0] * 0.95), int(face_input.shape[0] * 0.95)),
                                   (0, 255, 0),
                                   max(face_input.shape[0] // 100, 1))
        face_input = cv2.resize(face_input, (face_config.FACE_SIZE * 5, face_config.FACE_SIZE * 5))
        cv2.putText(img=face_input,
                    text=string_img,
                    org=(face_input.shape[0] // 15, face_input.shape[0] // 8),
                    fontFace=cv2.QT_FONT_NORMAL,
                    fontScale=0.75,
                    color=(0, 255, 0),
                    thickness=2)

        if imshow:
            cv2.imshow("face", face_input)

        if not video_mode:
            cv2.imwrite(config.OUTPUT_FOLDER_PATH + "grad_cam.jpg", cv2.cvtColor(result_npy, cv2.COLOR_RGB2BGR))
            cv2.imwrite(config.OUTPUT_FOLDER_PATH + "emotion.jpg", cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB))

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()

    return return_objs


def predict(
        image=None,
        video_mode=False,
        webcam_mode=False,
        model_save_path=config.FACE_MODEL_SAVE_PATH,
        best_hp_json_path=config.FACE_BEST_HP_JSON_SAVE_PATH,
        verbose=face_config.PREDICT_VERBOSE,
        imshow=face_config.SHOW_PRED_IMAGE,
        grad_cam=face_config.GRAD_CAM,
        grad_cam_on_video=face_config.GRAD_CAM_ON_VIDEO,
):
    """
    Predicts the emotion of the face in the image or video.
    Takes the full image, crops the face, detects the landmarks, and then runs the model on the face image and the landmarks.

    Parameters
    ----------
    image - path to image or video, or a numpy array of the image. Numpy array will only work if not video_mode or webcam_mode
        (Note: program currently only supports one face per image, if you'd like to add support for multiple faces, please submit a pull request.
        You'd just need to detect the faces using face_mesh.py or something similar, and then run the model on each face)
    video_mode - if True, will run the model on each frame of the video
    webcam_mode - if True, will run the model on each frame of the webcam
    model_save_path - path to the model to load
    imshow - if True, will show the image with the prediction
    verbose - if True, will print out the prediction probabilities

    Returns
    -------
    You'll get a tuple of the following based on the argments you pass in:

    if not webcam_mode and not video_mode:
        if grad_cam:
            Emotion name, emotion index, list of prediction probabilities, image as numpy, grad cam overlay as numpy
        else:
            Emotion name, emotion index, list of prediction probabilities, image as numpy
    else:
        None

    """

    best_hyperparameters = face_utils.load_dict_from_json(best_hp_json_path)
    if verbose:
        print(f"Best hyperparameters, {best_hyperparameters}")

    model = torch.load(model_save_path)
    model.to(config.device).eval()

    if video_mode:
        cap = cv2.VideoCapture(image)
        fps_in = cap.get(cv2.CAP_PROP_FPS)

        while True:
            init_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            _get_prediction(best_hp=best_hyperparameters, img=frame, model=model, imshow=False, video_mode=True, verbose=verbose)

            wait_time = round((1000 / fps_in) - (time.time() - init_time))
            print("wait_time: ", wait_time)
            cv2.waitKey(wait_time)

        return None
    if webcam_mode:
        cap = cv2.VideoCapture(0)
        while True:
            init_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            _get_prediction(best_hp=best_hyperparameters,
                            img=frame,
                            model=model,
                            imshow=True,
                            video_mode=True,
                            verbose=verbose,
                            grad_cam=True,
                            grad_cam_on_video=grad_cam_on_video,
                            feature_maps_flag=False)

            cv2.waitKey(1)

        return None
    else:
        if type(image) == str:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = _get_prediction(best_hp=best_hyperparameters, img=image, model=model, imshow=imshow, video_mode=video_mode, verbose=verbose, grad_cam=grad_cam)

        if verbose:
            print("\n\n\nResults:")
            for res in result:
                # check if numpy
                if type(res) == np.ndarray:
                    print(res.shape)
                else:
                    print(res)

        return result
