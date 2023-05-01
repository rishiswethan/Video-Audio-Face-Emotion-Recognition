
import source.audio_analysis_utils.utils as utils
import source.config as config
import source.audio_analysis_utils.audio_config as audio_config
import source.audio_analysis_utils.preprocess_data as data

import source.pytorch_utils.callbacks as pt_callbacks
import source.pytorch_utils.training_utils as pt_train
import source.pytorch_utils.hyper_tuner as pt_tuner

import torch
import torch.nn as nn
from torchvision.models import *
import numpy as np
import os
import time

from hyperopt import fmin, space_eval, Trials, rand

enable_validation = True
train_cnt = 0
tune_cnt = 0
total_tune_cnt = 0
start_time = 0

device = config.device

TUNE_TARGET = audio_config.TUNE_TARGET
TUNE_MODE = audio_config.TUNE_MODE
TUNER_CSV_SAVE_PATH = config.AUDIO_TUNER_CSV_SAVE_PATH
TUNER_SAVE_PATH = config.AUDIO_TUNER_SAVE_PATH
BEST_HP_JSON_SAVE_PATH = config.AUDIO_BEST_HP_JSON_SAVE_PATH
TUNE_HP_RANGES = audio_config.tune_hp_ranges
MAX_TRIALS = audio_config.max_trails

INITIAL_LR = audio_config.lr
INITIAL_EPOCH = audio_config.initial_epoch
REDUCE_LR_FACTOR = audio_config.reduce_lr_factor
REDUCE_LR_PATIENCE = audio_config.reduce_lr_patience
EARLY_STOPPING_PATIENCE = audio_config.patience_epoch
SOFTMAX_LEN = audio_config.softmax_len

MODEL_SAVE_PATH = config.AUDIO_MODEL_SAVE_PATH

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CustomModel(pt_train.CustomModelBase):
    """
    A custom model that inherits from CustomModelBase.
    This class is meant to be used with the training_utils.py module.

    Parameters
    ----------
    input_shape - The shape of the input data (n, 3, height, width) of the MFCCs
    dropout_rate - The dropout rate to use
    dense_units - The number of units in the dense layer
    num_layers - The number of dense layers
    l1_l2_reg - The L1 and L2 regularization to use (Not implemented yet)
    layers_batch_norm - Whether to use batch normalization in the dense layers
    conv_model_name - The name of the convolutional model to use. Choose from the list in the get_conv_model function
    class_weights : list - The class weights to use. If None, all classes will have the same weight
    device - The device to use

    """

    def __init__(self, input_shape, dropout_rate, dense_units, num_layers, l1_l2_reg, layers_batch_norm, conv_model_name, class_weights=None, device=device):
        if class_weights is None:
            class_weights = torch.ones(SOFTMAX_LEN)
        else:
            class_weights = torch.tensor(class_weights)

        # convert to cuda tensor
        class_weights = class_weights.to(device)

        super(CustomModel, self).__init__(class_weights)

        self.base_model = self.get_conv_model(conv_model_name)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        self.flatten = nn.Flatten()

        # Determine the output size of the base model
        with torch.no_grad():
            sample_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
            print("sample_input: ", sample_input.shape)
            base_output_size = self.base_model(sample_input).numel()
            print("base_output_size: ", base_output_size)

        # Calculate the input size for the dense layers
        dense_input_size = base_output_size

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

        self.fc = nn.Sequential(
            nn.Linear(dense_input_size, dense_units),  # Update the input size for the first dense layer
            nn.BatchNorm1d(dense_units),
            *dense_layers,
        )

        self.out = nn.Linear(dense_units, SOFTMAX_LEN)

    def get_conv_model(self, conv_model_name, pretrained=False):
        if conv_model_name == "resnet50":
            return resnet50(pretrained=pretrained)
        elif conv_model_name == "resnet18":
            return resnet18(pretrained=pretrained)
        elif conv_model_name == "resnet34":
            return resnet34(pretrained=pretrained)
        elif conv_model_name == "resnet101":
            return resnet101(pretrained=pretrained)
        elif conv_model_name == "resnet152":
            return resnet152(pretrained=pretrained)
        elif conv_model_name == "resnext50_32x4d":
            return resnext50_32x4d(pretrained=pretrained)
        elif conv_model_name == "resnext101_32x8d":
            return resnext101_32x8d(pretrained=pretrained)
        elif conv_model_name == "wide_resnet50_2":
            return wide_resnet50_2(pretrained=pretrained)
        elif conv_model_name == "wide_resnet101_2":
            return wide_resnet101_2(pretrained=pretrained)
        elif conv_model_name == "inception":
            raise NotImplementedError
            # not working as of now
            model = inception_v3(pretrained=pretrained)
            model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            return model
        elif conv_model_name == "googlenet":
            return googlenet(pretrained=pretrained)
        elif conv_model_name == "mobilenet":
            return mobilenet_v2(pretrained=pretrained)
        elif conv_model_name == "densenet":
            return densenet121(pretrained=pretrained)
        elif conv_model_name == "alexnet":
            return alexnet(pretrained=pretrained)
        elif conv_model_name == "vgg16":
            return vgg16(pretrained=pretrained)
        elif conv_model_name == "squeezenet":
            return squeezenet1_0(pretrained=pretrained)
        elif conv_model_name == "shufflenet":
            return shufflenet_v2_x1_0(pretrained=pretrained)
        elif conv_model_name == "mnasnet":
            return mnasnet1_0(pretrained=pretrained)
        else:
            raise ValueError("Invalid model name, exiting...")

    def forward(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)

        return x


def get_callbacks(
        optimiser: torch.optim.Optimizer,
        result: dict,
        model: pt_train.CustomModelBase,
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
def train(hp_dict, metric='val_acc', metric_mode='max', preprocess_again=False, initial_lr=INITIAL_LR, epochs=INITIAL_EPOCH):
    """

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

    N_FFT = hp_dict['N_FFT']
    HOP_LENGTH = hp_dict['HOP_LENGTH']
    NUM_MFCC = hp_dict['NUM_MFCC']

    batch_size = hp_dict['batch_size']
    dropout_rate = hp_dict['dropout_rate']
    dense_units = hp_dict['dense_units']
    num_layers = hp_dict['num_layers']
    l1_l2_reg = hp_dict['l1_l2_reg']
    conv_model = hp_dict['conv_model']
    layers_batch_norm = bool(hp_dict['layers_batch_norm'])

    preprocessor_string = f'N_FFT={N_FFT}_HOP_LENGTH={HOP_LENGTH}_NUM_MFCC={NUM_MFCC}'

    # Load saved preprocessed data
    X, Y, classification_Y = data.preprocess_audio(
        NUM_MFCC=NUM_MFCC,
        N_FFT=N_FFT,
        HOP_LENGTH=HOP_LENGTH,
        save_name_prefix=preprocessor_string,
        force_preprocess_files=preprocess_again,
        force_clean_files=preprocess_again,
        print_flag=True
    )

    # Split the data into train and test
    X, Y, Xt, Yt = data.split_data(X=X, Y=Y, classifications=classification_Y, test_split_percent=audio_config.test_split_percentage)

    print("X.shape: ", X.shape)
    print("Y.shape: ", Y.shape)
    print("Xt.shape: ", Xt.shape)
    print("Yt.shape: ", Yt.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3)
    Xt = Xt.reshape(Xt.shape[0], Xt.shape[1], Xt.shape[2], 3)

    input_shape = utils.get_input_shape(target_hp=hp_dict)

    class_weights = utils.get_class_weights(Y)
    class_weights = list(class_weights.values())
    print(f"Class weights: {class_weights}")

    # transpose the data to be in the format of (num_images, channels, height, width) from (num_images, height, width, channels)
    X = np.transpose(X, (0, 3, 1, 2))
    Xt = np.transpose(Xt, (0, 3, 1, 2))
    print(f"\n\ninput_shape: {input_shape}")
    print(f"X_images.shape: {X.shape}")
    print(f"Xt shape: {(Xt.shape)}, Yt shape: {Yt.shape}")

    # Convert the data to torch tensors
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y)
    Xt = torch.from_numpy(Xt).float()
    Yt = torch.from_numpy(Yt)

    # Convert the data to torch datasets
    train_dataset = DataGenerator(X, Y)
    val_dataset = DataGenerator(Xt, Yt)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    model = CustomModel(
        input_shape=input_shape,
        class_weights=class_weights,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        num_layers=num_layers,
        l1_l2_reg=l1_l2_reg,
        layers_batch_norm=layers_batch_norm,
        conv_model_name=conv_model
    )
    # summary(model, [input_shape_1, input_shape_2], device=device)

    # Train the model using torch
    history = pt_train.fit(
        epochs=epochs,
        lr=initial_lr,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks_function=get_callbacks,
    )

    if metric and metric_mode:
        acc_min, acc_max = get_min_max_vale(history, metric)
        opt_result = acc_min if metric_mode == 'min' else acc_max

        # set to - if metric_mode is min, else set to +. This is for hyperopt to work
        opt_result = -opt_result if metric_mode == 'min' else opt_result

        return opt_result


def train_using_best_values(best_hp_json_save_path=BEST_HP_JSON_SAVE_PATH, preprocess_again=False):
    """
    Train the model using the best hyperparameters found by hyperparameter optimisation
    Parameters
    ----------
    best_hp_json_save_path - path to the json file containing the best hyperparameters
    preprocess_again - whether to preprocess the data again or not

    """

    best_hyperparameters = utils.load_dict_from_json(best_hp_json_save_path)
    print(f"Best hyperparameters, {best_hyperparameters}")

    train(hp_dict=best_hyperparameters, preprocess_again=preprocess_again)


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
    )

    tuner_utils.start_time = time.time()

    # Get the hp objects for each range in hyperopt
    search_space_hyperopt = tuner_utils.return_full_hp_dict(search_space)
    trials = Trials()

    best = fmin(
        tuner_utils.train_for_tuning,
        search_space_hyperopt,
        algo=rand.suggest,
        max_evals=tuner_utils.max_trials,
        trials=trials,
        trials_save_file=tuner_utils.tuner_obj_save_path,
        verbose=True,
        show_progressbar=False
    )

    print("Best: ", best)
    print(space_eval(search_space_hyperopt, best))

    # pt_utils.hyper_tuner class will save the best hyperparameters to a json file after each trial


class DataGenerator(torch.utils.data.Dataset):
    """
    Simple data generator to load the data into the model
    """
    def __init__(self, X_images, Y):
        self.X_images = X_images
        self.Y = Y

    def __len__(self):
        return len(self.X_images)

    def __getitem__(self, index):
        X_image = self.X_images[index]
        Y = self.Y[index]

        return X_image, Y


if __name__ == '__main__':
    # train_using_best_values()
    hyper_parameter_optimise()
