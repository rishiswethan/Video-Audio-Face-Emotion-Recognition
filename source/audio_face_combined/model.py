# importing face_emotion_utils
import source.face_emotion_utils.model as face_model
import source.face_emotion_utils.face_config as face_config

# importing combined_emotion_utils
import source.audio_face_combined.preprocess_main as combined_data
import source.audio_face_combined.utils as combined_utils
import source.audio_face_combined.combined_config as combined_config

# importing audio_analysis_utils
import source.audio_analysis_utils.utils as audio_utils
import source.audio_analysis_utils.model as audio_model

# importing pytorch_utils
import source.pytorch_utils.callbacks as pt_callbacks
import source.pytorch_utils.training_utils as pt_train
import source.pytorch_utils.hyper_tuner as pt_tuner

import source.config as config

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from hyperopt import fmin, tpe, space_eval, Trials, rand
import numpy as np
import albumentations as albu


enable_validation = True
train_cnt = 0
tune_cnt = 0
total_tune_cnt = 0
start_time = 0

# tune_hp_ranges = {}
device = config.device

TUNE_TARGET = combined_config.TUNE_TARGET
TUNE_MODE = combined_config.TUNE_MODE
TUNER_CSV_SAVE_PATH = config.AUDIO_FACE_COMBINED_TUNER_CSV_SAVE_PATH
TUNER_SAVE_PATH = config.AUDIO_FACE_COMBINED_TUNER_SAVE_PATH
BEST_HP_JSON_SAVE_PATH = config.AUDIO_FACE_COMBINED_BEST_HP_JSON_SAVE_PATH
TUNE_HP_RANGES = combined_config.tune_hp_ranges
MAX_TRIALS = combined_config.max_trails

INITIAL_LR = combined_config.lr
INITIAL_EPOCH = combined_config.initial_epoch
REDUCE_LR_FACTOR = combined_config.reduce_lr_factor
REDUCE_LR_PATIENCE = combined_config.reduce_lr_patience
EARLY_STOPPING_PATIENCE = combined_config.early_stopping_patience_epoch
SOFTMAX_LEN = combined_config.softmax_len

FACE_SIZE = face_config.FACE_SIZE

FACE_MODEL_SAVE_PATH = config.FACE_MODEL_SAVE_PATH
AUDIO_MODEL_SAVE_PATH = config.AUDIO_MODEL_SAVE_PATH
MODEL_SAVE_PATH = config.AUDIO_FACE_COMBINED_MODEL_SAVE_PATH

FACE_BEST_HP_JSON_SAVE_PATH = config.FACE_BEST_HP_JSON_SAVE_PATH
AUDIO_BEST_HP_JSON_SAVE_PATH = config.AUDIO_BEST_HP_JSON_SAVE_PATH

MAX_THREADS = config.MAX_THREADS

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CustomModelBase(pt_train.CustomModelBase):
    """
    ModelBase override for training and validation steps
    """
    def __init__(self, class_weights):
        super(CustomModelBase, self).__init__()
        self.class_weights = class_weights

    def training_step(self, batch):
        all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X, all_video_emotions_Y = batch
        out = self(all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X)  # Generate predictions

        loss = F.cross_entropy(out, all_video_emotions_Y, weight=self.class_weights)  # Calculate loss with class weights
        acc = pt_train.accuracy(out, all_video_emotions_Y)  # Calculate accuracy
        return loss, acc

    def validation_step(self, batch):
        all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X, all_video_emotions_Y = batch
        out = self(all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X)  # Generate predictions

        loss = F.cross_entropy(out, all_video_emotions_Y, weight=self.class_weights)  # Calculate loss with class weights
        acc = pt_train.accuracy(out, all_video_emotions_Y)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}


class CustomModel(CustomModelBase):
    """
    Custom model for audio and face combined model.
    The sequence of inputs for the face model, taken from the frames of the video, the audio model input and the text sentiment model input are concatenated and passed through a dense layer.
    The sequence of inputs are passed through LSTM layer(s), and then combined with the dense models of the face and audio models, and the text sentiment output.
    Dense layers are defined after the combined results from above to a final softmax layer.

    Args:
        audio_model (tuple): (audio model, number of outputs of the last layer of the audio model before the output layer)
        image_model (tuple): (image model, number of outputs of the last layer of the image model before the output layer)
        num_images (int): Number of images to be passed to the image model in a sequence.
        input_shapes (tuple): Tuple of input shapes for the 3 models. (audio, image, text)
        dropout_rate (float): Dropout rate for the dense layers.
        dense_units (int): Number of units in the dense layers.
        num_layers (int): Number of dense layers.
        l1_l2_reg (float): L1 and L2 regularization factor.(Not implemented)
        layers_batch_norm (bool): Whether to use batch normalization in the dense layers.
        sequence_model_dense_units (int): Number of units in the dense layers of the sequence model.
        sequence_model_layers (int): Number of dense layers of the sequence model.
        softmax_len (int, optional): Number of classes. Defaults to SOFTMAX_LEN.
        sequence_model (bool, optional): Whether to use sequence model. Defaults to True.
        class_weights (list, optional): List of class weights. Defaults to all ones.

    """
    def __init__(self,
                 audio_model,
                 image_model,
                 num_images,
                 input_shapes,
                 dropout_rate,
                 dense_units,
                 num_layers,
                 l1_l2_reg,
                 layers_batch_norm,
                 sequence_model_dense_units,
                 sequence_model_layers,
                 softmax_len=SOFTMAX_LEN,
                 sequence_model=True,
                 class_weights=None):

        if class_weights is None:
            class_weights = torch.ones(softmax_len)
        else:
            class_weights = torch.tensor(class_weights)

        # convert to cuda tensor
        class_weights = class_weights.to(device)

        super(CustomModel, self).__init__(class_weights=class_weights)

        self.audio_model, audio_last_layer_num = audio_model
        self.image_model, face_last_layer_num = image_model
        self.num_images = num_images
        self.sequence_model = sequence_model

        self.class_weights = class_weights

        self.dropout = nn.Dropout(dropout_rate)

        self.batch_norm_image = nn.BatchNorm1d(face_last_layer_num)
        self.batch_norm_audio = nn.BatchNorm1d(audio_last_layer_num)

        # figure out the combined input size from the last layer of the image and audio models and the text model
        dense_input_size_post_comb = sequence_model_dense_units + audio_last_layer_num + input_shapes[2]

        dense_layers = []
        for _ in range(num_layers - 1):  # Update the range to num_layers - 1
            dense_layer = [
                nn.Linear(dense_units, dense_units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            if layers_batch_norm:
                dense_layer.append(nn.BatchNorm1d(dense_units))
            dense_layers.extend(dense_layer)

        if sequence_model:
            self.lstm_layers = nn.LSTM(input_size=face_last_layer_num, hidden_size=sequence_model_dense_units, num_layers=sequence_model_layers, dropout=dropout_rate, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(dense_input_size_post_comb, dense_units),  # Update the input size for the first dense layer
            nn.BatchNorm1d(dense_units),
            *dense_layers
        )

        self.output_layer = nn.Linear(dense_units, softmax_len)

    # noinspection PyCallingNonCallable
    def forward(self, input_image_X1, input_image_X2, input_audio_X, input_text_X):
        # convert single channel image to 3 channel image

        # Add a new dimension to the tensor, making it (n, 1, 64, 64)
        input_audio_X = input_audio_X.unsqueeze(1)
        # Repeat the single channel 3 times along the second axis, resulting in (n, 3, 64, 64)
        input_audio_X = torch.repeat_interleave(input_audio_X, repeats=3, dim=1)

        output_audio = self.audio_model(input_audio_X)
        output_audio = self.dropout(output_audio)
        output_audio = self.batch_norm_audio(output_audio)

        image_outputs = []
        for i in range(self.num_images):
            output_image = self.image_model(input_image_X1[:, i], input_image_X2[:, i])
            output_image = self.dropout(output_image)
            output_image = self.batch_norm_image(output_image)
            image_outputs.append(output_image.unsqueeze(1))

        image_sequence = torch.cat(image_outputs, dim=1)

        if self.sequence_model:
            output_sequence, _ = self.lstm_layers(image_sequence)
            output_sequence = output_sequence[:, -1, :]
        else:
            output_sequence = torch.cat(image_outputs, dim=-1)

        output_concat = torch.cat([output_audio, output_sequence, input_text_X], dim=-1)

        X = self.fc(output_concat)
        X = self.output_layer(X)

        return X


class FaceModelNoSoftmax(face_model.CustomModel):
    """
    An override of the CustomModel class to remove the softmax layer is required, as torch.nn.Sequential(*list(face_model_defined.children())[:-1]) does not work
    when we need to give it multiple inputs, although it works fine when we give it a single input as in the audio model.
    """

    def __init__(self, face_model_save_path, **kwargs):
        super(FaceModelNoSoftmax, self).__init__(**kwargs)

        if os.path.exists(face_model_save_path):
            loaded_model = torch.load(face_model_save_path)
            self.load_state_dict(loaded_model.state_dict())
            print("\nFace model loaded\n")
        else:
            input("Face model not found. Press enter to continue training with random weights.")

    def forward(self, x1, x2):
        x1 = self.base_model(x1)
        x1 = self.flatten(x1)

        if self.use_landmarks:
            x = torch.cat((x1, x2), dim=1)
            x = self.fc(x)
        else:
            x = x1

        return x


class AudioModelNoSoftmax(audio_model.CustomModel):
    """
    An override of the CustomModel class to remove the softmax layer
    """

    def __init__(self, audio_model_save_path, **kwargs):
        super(AudioModelNoSoftmax, self).__init__(**kwargs)

        if os.path.exists(audio_model_save_path):
            loaded_model = torch.load(audio_model_save_path)
            self.load_state_dict(loaded_model.state_dict())
            print("\nAudio model loaded\n")
        else:
            input("Audio model not found. Press enter to continue training with random weights.")

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.fc(x)
        # skip the softmax layer

        return x


def get_face_model(best_hp_json_save_path=FACE_BEST_HP_JSON_SAVE_PATH, face_model_save_path=FACE_MODEL_SAVE_PATH):
    """
    Load the saved face model with the best hyperparameters

    Parameters
    ----------
    best_hp_json_save_path: str
        The path to the json file containing the best hyperparameters for the face model
    face_model_save_path: str
        The path to the saved face model

    Returns
    -------
    face_model_defined: FaceModelNoSoftmax
        The face model with the best hyperparameters
    last_feature_num: int
        The number of features in the last layer of the face model, before the softmax layer

    """
    hp_dict = audio_utils.load_dict_from_json(best_hp_json_save_path)

    dropout_rate = hp_dict['dropout_rate']
    dense_units = hp_dict['dense_units']
    num_layers = hp_dict['num_layers']
    l1_l2_reg = hp_dict['l1_l2_reg']
    layers_batch_norm = bool(hp_dict['layers_batch_norm'])
    conv_model = hp_dict['conv_model']
    batch_size = hp_dict['batch_size']
    use_landmarks = bool(hp_dict['use_landmarks'])

    input_shape_1 = combined_utils.get_input_shape(which_input='image')
    input_shape_2 = combined_utils.get_input_shape(which_input='landmarks_depths')

    face_model_defined = FaceModelNoSoftmax(
        face_model_save_path=face_model_save_path,
        input_shapes=(input_shape_1, input_shape_2),
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        num_layers=num_layers,
        l1_l2_reg=l1_l2_reg,
        layers_batch_norm=layers_batch_norm,
        conv_model_name=conv_model,
        use_landmarks=use_landmarks
    ).to(device).train()

    with torch.no_grad():
        # create 2 random inputs of these sizes
        # torch.Size([256, 3, 64, 64]), torch.Size([256, 1404])
        input_image_X1 = torch.rand(batch_size, input_shape_1[0], input_shape_1[1], input_shape_1[2]).to(device)
        input_image_X2 = torch.rand(batch_size, input_shape_2).to(device)

        # run the model once to initialize the weights
        output = face_model_defined(input_image_X1, input_image_X2)
        last_feature_num = output.shape[1]
        print("\nimage output: ", output.shape)
        print("last image: ", last_feature_num, "\n")

    return face_model_defined, last_feature_num


def get_audio_model(best_hp_json_save_path=AUDIO_BEST_HP_JSON_SAVE_PATH, audio_model_save_path=AUDIO_MODEL_SAVE_PATH):
    """
    Load the saved audio model with the best hyperparameters

    Parameters
    ----------
    best_hp_json_save_path: str
        The path to the json file containing the best hyperparameters for the audio model
    audio_model_save_path: str
        The path to the saved audio model

    Returns
    -------
    audio_model_defined: AudioModelNoSoftmax
        The audio model with the best hyperparameters
    last_feature_num: int
        The number of features in the last layer of the audio model, before the softmax layer

    """
    hp_dict = audio_utils.load_dict_from_json(best_hp_json_save_path)

    dropout_rate = hp_dict['dropout_rate']
    dense_units = hp_dict['dense_units']
    num_layers = hp_dict['num_layers']
    l1_l2_reg = hp_dict['l1_l2_reg']
    layers_batch_norm = bool(hp_dict['layers_batch_norm'])
    conv_model = hp_dict['conv_model']
    batch_size = hp_dict['batch_size']

    input_shape = combined_utils.get_input_shape(which_input='audio')

    # Create the model
    audio_model_defined = AudioModelNoSoftmax(
        audio_model_save_path=audio_model_save_path,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        num_layers=num_layers,
        l1_l2_reg=l1_l2_reg,
        layers_batch_norm=layers_batch_norm,
        conv_model_name=conv_model
    ).to(device).train()

    with torch.no_grad():
        # create a random input of this size
        audio_input_X = torch.rand(batch_size, input_shape[0], input_shape[1], input_shape[2]).to(device)
        output = audio_model_defined(audio_input_X)
        last_layer_num_features = output.shape[1]
        print("\naudio output: ", output.shape)
        print("last audio: ", last_layer_num_features, "\n")

    return audio_model_defined, last_layer_num_features


# This function is to define the 3 models that we combine
def def_all_models(num_images, h_params, load_audio_image_model=True, class_weights=None):
    """
    Define the 2 models that we combine in the combined model
    Parameters
    ----------
    num_images - the number of images per datapoint that goes into the sequence layers of the combined model
    h_params - the hyperparameters for the combined model
    load_audio_image_model - whether to load the audio and image models from the saved models
    class_weights: list - the class weights for the combined model

    Returns
    -------
    face_model_defined - the face model with the best hyperparameters

    """
    comb_h_params = h_params.copy()
    previous_layers_trainable = bool(comb_h_params['prev_layers_trainable'])
    dropout_rate_comb = comb_h_params['dropout_rate']
    l1_l2_reg_comb = comb_h_params['l1_l2_reg']
    dense_units_comb = comb_h_params['dense_units']
    num_layers_comb = comb_h_params['num_layers']
    sequence_model_dense_units = comb_h_params['sequence_model_dense_units']
    sequence_model_layers = comb_h_params['sequence_model_layers']
    layers_batch_norm = bool(comb_h_params['layers_batch_norm'])

    # Get the input shapes
    input_shape_image = combined_utils.get_input_shape(which_input='image')
    input_shape_lands = combined_utils.get_input_shape(which_input='landmarks_depths')
    input_shape_audio = combined_utils.get_input_shape(which_input='audio')
    input_shape_text_sent = combined_utils.get_input_shape(which_input='text_sentiment')

    # Get the face and audio models and the number of features in the last layer before the softmax layer
    face_model_defined, face_last_layer_num = get_face_model()
    audio_model_defined, audio_last_layer_num = get_audio_model()

    h_params = comb_h_params
    print(f"Combined best hyperparameters, {h_params}")

    # Define combined model
    input_shapes = ((input_shape_image, input_shape_lands), input_shape_audio, input_shape_text_sent)
    print("Input shapes", input_shapes)
    combined_model = CustomModel(image_model=(face_model_defined, face_last_layer_num),
                                 audio_model=(audio_model_defined, audio_last_layer_num),
                                 input_shapes=input_shapes,
                                 class_weights=class_weights,
                                 num_images=num_images,
                                 dropout_rate=dropout_rate_comb,
                                 dense_units=dense_units_comb,
                                 num_layers=num_layers_comb,
                                 l1_l2_reg=l1_l2_reg_comb,
                                 layers_batch_norm=layers_batch_norm,
                                 sequence_model_dense_units=sequence_model_dense_units,
                                 sequence_model_layers=sequence_model_layers).to(device).train()

    return combined_model, face_model_defined, audio_model_defined


def get_callbacks(
        optimiser,
        result,
        model,
        defined_callbacks=None,
        reduce_lr_factor=REDUCE_LR_FACTOR,
        reduce_lr_patience=REDUCE_LR_PATIENCE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
):
    """
    Parameters
    ----------
    optimiser: torch.optim.Optimizer
    result: dict
        dictionary with keys 'train_loss', 'val_acc', 'train_acc', 'val_loss', and any other metrics you want to use
    model: pt_train.CustomModelBase
        Model must override the CustomModelBase class
    defined_callbacks
        Default is None. If None, then the default callbacks will be used.
    reduce_lr_factor: float
    reduce_lr_patience: int
    early_stopping_patience: int

    Returns
    -------
    defined_callbacks: dict of pt_callbacks.Callbacks

    step_flag: bool
        True if the training should stop, False otherwise, based on the early stopping callback
    """

    if defined_callbacks is None:
        defined_callbacks = {
            'val': pt_callbacks.Callbacks(model_save_path=MODEL_SAVE_PATH),
            'train': pt_callbacks.Callbacks()
        }

    defined_callbacks['val'].model_checkpoint(
        model=model,
        monitor_value=result['val_acc'],
        mode='max',
        indicator_text="Val checkpoint: "
    )
    defined_callbacks['train'].reduce_lr_on_plateau(
        optimizer=optimiser,
        monitor_value=result['train_loss'],
        mode='min',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        indicator_text="Train LR scheduler: "
    )
    stop_flag = defined_callbacks['val'].early_stopping(
        monitor_value=result['val_acc'],
        mode='max',
        patience=early_stopping_patience,
        indicator_text="Val early stopping: "
    )
    defined_callbacks['train'].clear_memory()
    print("_________")

    return defined_callbacks, stop_flag


# Once the best hyperparameters are found using tune_hyperparameters(), call this function to train the model
def train(hp_dict, metric='val_acc', metric_mode='max', preprocess_again=False, initial_lr=INITIAL_LR, epochs=INITIAL_EPOCH, continue_training=True, max_threads=combined_config.MAX_TRAINING_THREADS):
    """
    Once the best hyperparameters are found using tune_hyperparameters(), call this function to train the model with the best hyperparameters found.

    Parameters
    ----------
    hp_dict: dict
        Contains the hyperparameters to be used for training and preprocessing.

    metric: str
        Target metric whose max or min value is to be found in the training process and returned. Will be used to find the best hyperparameters.

    metric_mode: str
        'max' or 'min' depending on whether the metric is to be maximised or minimised

    preprocess_again: bool
        If True, the data will be preprocessed again. If False, the data will be loaded from the preprocessed files.

    initial_lr: float
        Initial learning rate to be used for training. Can be scheduled to change during training using the reduce_lr_on_plateau callback in the pytorch_callbacks.py file.

    epochs: int
        Number of epochs to train for. Can step out of the training loop early if the early_stopping callback in the pytorch_callbacks.py file is triggered.

    Returns
    -------
    opt_result: float
        The best value of the metric found during training. This is the value that will be used to find the best hyperparameters.

    """
    def get_min_max_vale(history, key):
        min = 99999
        max = -99999
        for i in range(len(history)):
            if history[i][key] < min:
                min = history[i][key]
            if history[i][key] > max:
                max = history[i][key]

        return min, max

    print("\nTraining with hyperparameters: ")
    for key, value in hp_dict.items():
        print(f"{key}: {value}")

    # Train hyperparameters
    batch_size = hp_dict['batch_size']

    # Load saved preprocessed data and split into train and test
    train_data, test_data = combined_data.split_data(test_split_percent=combined_config.test_split_percentage)
    (all_frames_face_images_X_train, all_frames_face_lands_X_train, all_extracted_mfcc_list_train, all_text_sentiment_X_train, all_video_emotions_Y_train) = train_data
    (all_frames_face_images_X_test, all_frames_face_lands_X_test, all_extracted_mfcc_list_test, all_text_sentiment_X_test, all_video_emotions_Y_test) = test_data

    # calculate the class weights
    class_weights = audio_utils.get_class_weights(all_video_emotions_Y_train)
    class_weights = list(class_weights.values())
    print(f"Class weights: {class_weights}")

    print(f"all_frames_face_images_X_train.shape: {all_frames_face_images_X_train.shape}")
    print(f"all_frames_face_lands_X_train.shape: {all_frames_face_lands_X_train.shape}")
    print(f"all_extracted_mfcc_list_train.shape: {all_extracted_mfcc_list_train.shape}")
    print(f"all_text_sentiment_X_train.shape: {all_text_sentiment_X_train.shape}")
    print(f"all_video_emotions_Y_train.shape: {all_video_emotions_Y_train.shape}")

    print(f"all_frames_face_images_X_test.shape: {all_frames_face_images_X_test.shape}")
    print(f"all_frames_face_lands_X_test.shape: {all_frames_face_lands_X_test.shape}")
    print(f"all_extracted_mfcc_list_test.shape: {all_extracted_mfcc_list_test.shape}")
    print(f"all_text_sentiment_X_test.shape: {all_text_sentiment_X_test.shape}")
    print(f"all_video_emotions_Y_test.shape: {all_video_emotions_Y_test.shape}")

    # convert train data to torch tensors
    # all_frames_face_images_X_train = torch.from_numpy(all_frames_face_images_X_train).float()
    all_frames_face_lands_X_train = torch.from_numpy(all_frames_face_lands_X_train).float()
    all_extracted_mfcc_list_train = torch.from_numpy(all_extracted_mfcc_list_train).float()
    all_text_sentiment_X_train = torch.from_numpy(all_text_sentiment_X_train).float()
    all_video_emotions_Y_train = torch.from_numpy(all_video_emotions_Y_train)
    # convert test data to torch tensors
    all_frames_face_images_X_test = torch.from_numpy(all_frames_face_images_X_test).float()
    all_frames_face_lands_X_test = torch.from_numpy(all_frames_face_lands_X_test).float()
    all_extracted_mfcc_list_test = torch.from_numpy(all_extracted_mfcc_list_test).float()
    all_text_sentiment_X_test = torch.from_numpy(all_text_sentiment_X_test).float()
    all_video_emotions_Y_test = torch.from_numpy(all_video_emotions_Y_test)

    # get the defined model
    combined_model, face_model_defined, audio_model_defined = def_all_models(
        num_images=all_frames_face_images_X_train.shape[1],
        h_params=hp_dict,
        class_weights=class_weights,
    )

    # Set augmentation only if not using landmarks, as the landmarks are not augmented
    use_landmarks = face_model_defined.use_landmarks
    print(f"use_landmarks variable: {use_landmarks}")
    augmentation = None if use_landmarks else get_training_augmentation
    num_workers = max_threads

    # Convert the data to torch datasets
    train_dataset = DataGenerator(all_frames_face_images_X_train, all_frames_face_lands_X_train, all_extracted_mfcc_list_train, all_text_sentiment_X_train, all_video_emotions_Y_train, augmentation)
    val_dataset = DataGenerator(all_frames_face_images_X_test, all_frames_face_lands_X_test, all_extracted_mfcc_list_test, all_text_sentiment_X_test, all_video_emotions_Y_test, None)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Train the model using torch
    history = pt_train.fit(
        epochs=epochs,
        lr=initial_lr,
        model=combined_model,
        train_loader=train_loader,
        callbacks_function=get_callbacks,
        val_loader=val_loader
    )

    if metric and metric_mode:
        acc_min, acc_max = get_min_max_vale(history, metric)
        opt_result = acc_min if metric_mode == 'min' else acc_max

        # set to - if metric_mode is min, else set to +. This is for hyperopt to work
        opt_result = -opt_result if metric_mode == 'min' else opt_result

        return opt_result


def train_using_best_values(best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH):
    """
    Train the model using the best hyperparameters found by hyperparameter optimisation
    Parameters
    ----------
    best_hp_json_save_path - path to the json file containing the best hyperparameters
    preprocess_again - whether to preprocess the data again or not

    """
    best_hyperparameters = audio_utils.load_dict_from_json(best_hp_json_save_path)
    print(f"Best hyperparameters, {best_hyperparameters}")

    train(hp_dict=best_hyperparameters)


def hyper_parameter_optimise(
        search_space=TUNE_HP_RANGES,
        best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH,
        tuner_csv_save_path=TUNER_CSV_SAVE_PATH,
        tuner_obj_save_path=TUNER_SAVE_PATH,
        tune_target=TUNE_TARGET,
        max_trials=MAX_TRIALS,
        load_if_exists=True,
):
    """
    Main function for hyperparameter optimisation using hyperopt

    Parameters
    ----------
    search_space: dict
        Example:
            tune_hp_ranges = {
                "dropout_rate": ([0.0, 0.3, 4], 'range')
                "conv_model": (["resnet18", "resnet101", "resnext50_32x4d"], 'choice'),
            }

    best_hp_json_save_path: str
        Path to the json file where the best hyperparameters will be saved

    tuner_csv_save_path: str
        Path to the csv file where the hyperparameter tuning results will be saved.
        A modified version of the csv file will be saved in the same directory for sorted results

    tuner_obj_save_path: str
        Path to the file where the hyperparameter tuning object will be saved

    tune_target: str
        The metric to be optimised. This is the metric that will be used to find the best hyperparameters

    max_trials: int
        The maximum number of trials to be run for hyperparameter optimisation

    load_if_exists: bool
        Whether to load the tuner object from the tuner_obj_save_path if it exists or not.

    """

    global tune_cnt, total_tune_cnt, start_time

    if load_if_exists:
        print(f"Loading existing tuner object from {tuner_obj_save_path}")
    else:
        print(f"Creating new tuner object")

    tuner_utils = pt_tuner.HyperTunerUtils(
        best_hp_json_save_path=best_hp_json_save_path,
        tuner_csv_save_path=tuner_csv_save_path,
        tuner_obj_save_path=tuner_obj_save_path,
        tune_target=tune_target,
        tune_hp_ranges=search_space,
        max_trials=max_trials,
        train_function=train,
        load_if_exists=load_if_exists,
        seed=0
    )

    tuner_utils.start_time = time.time()

    # Get the hp objects for each range in hyperopt
    search_space_hyperopt = tuner_utils.return_full_hp_dict(search_space)
    trials = Trials()

    best = fmin(
        tuner_utils.train_for_tuning,
        search_space_hyperopt,
        algo=tuner_utils.suggest_grid,
        max_evals=tuner_utils.max_trials,
        trials=trials,
        trials_save_file=tuner_utils.tuner_obj_save_path,
        verbose=True,
        show_progressbar=False
    )

    print("Best: ", best)
    print(space_eval(search_space_hyperopt, best))

    # Our pt_utils.hyper_tuner class will save the best hyperparameters to a json file after each trial


def get_training_augmentation(height, width):
    def _get_training_augmentation(height, width):
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            # albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
            # albu.RandomCrop(height=height, width=width, always_apply=True),

            albu.IAAAdditiveGaussianNoise(p=0.2),
            # albu.IAAPerspective(p=0.5),
            albu.Rotate(limit=180, p=0.9),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    return _get_training_augmentation(height, width)


class DataGenerator(torch.utils.data.Dataset):
    """
    Simple data generator to load the data into the model
    """
    def __init__(self, all_frames_face_images_X, all_frames_face_lands_X, all_extracted_mfcc_list, all_text_sentiment_X, all_video_emotions_Y, image_augmentation=get_training_augmentation):
        self.X_images = all_frames_face_images_X
        self.X_landmark_depth = all_frames_face_lands_X
        self.X_mfcc = all_extracted_mfcc_list
        self.X_text_sentiment = all_text_sentiment_X
        self.Y = all_video_emotions_Y
        if image_augmentation:
            self.image_augmentation = image_augmentation(height=FACE_SIZE, width=FACE_SIZE)
            print("Image augmentation is enabled (combined_model.py)")
        else:
            self.image_augmentation = None
            print("Image augmentation is disabled (combined_model.py)")

    def __len__(self):
        return len(self.X_images)

    def __getitem__(self, index):
        X_image = self.X_images[index]
        X_landmark_depth = self.X_landmark_depth[index]
        X_mfcc = self.X_mfcc[index]
        X_text_sentiment = self.X_text_sentiment[index]
        Y = self.Y[index]

        if self.image_augmentation:
            # X_image = self.convert_tensor_to_numpy(X_image)
            X_image = X_image * 255.
            X_image = X_image.astype(np.uint8)

            X_image = np.transpose(X_image, (0, 2, 3, 1))

            images_aug = []
            for image in X_image:
                image_ = self.image_augmentation(image=image)['image']
                images_aug.append(image_)
            X_image = np.array(images_aug)

            X_image = X_image / 255.
            X_image = self.convert_numpy_to_tensor(X_image)

        return X_image, X_landmark_depth, X_mfcc, X_text_sentiment, Y

    def convert_tensor_to_numpy(self, tensor):
        img = tensor.detach().cpu().numpy()
        img = np.transpose(img, (0, 2, 3, 1))

        return img

    def convert_numpy_to_tensor(self, numpy_array):
        numpy_array = np.transpose(numpy_array, (0, 3, 1, 2))
        img = torch.from_numpy(numpy_array).to(device).type(torch.FloatTensor)

        return img
