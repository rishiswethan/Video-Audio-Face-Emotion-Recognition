import source.config as config


FACE_SIZE = 64  # This is the size of the face image that will be used for training
FACE_FOR_LANDMARKS_IMAGE_RESIZE = 512
LANDMARK_COMBINATIONS_DEPTHS_CNT = 1404

# PERMANENT HPARAMS
# Model hyperparameters
initial_epoch = 50000
lr = 0.001
patience_epoch = 25  # This is used during early stopping

reduce_lr_factor = 0.5
reduce_lr_patience = 5

# Data hyperparameters
test_split_percentage = 0.15
max_attime_data_pts = 100000 # This is used to prevent overloading RAM by saving data at x intervals

"""
Example hp range syntax:
Syntax:
"hp_name": ([hp_value1, hp_value2, ...], 'choice')
"hp_name": ([hp_range_start, hp_range_end, num_random_values], 'range') 

Examples:
"dropout_rate": ([0.0, 0.3, 4], 'range')  # 4 random values between 0.0 and 0.3
"dense_units": ([128, 256, 512], 'choice')  # 3 values to choose from
"""

tune_hp_ranges = {
    "batch_size": ([256], 'choice'),
    "dropout_rate": ([0.0], 'choice'),
    "l1_l2_reg": ([0.0],  'choice'),
    "dense_units": ([32], 'choice'),
    "num_layers": ([3], 'choice'),
    "normalise": ([True], 'choice'),
    "layers_batch_norm": ([True], 'choice'),
    "prev_layers_trainable": ([True], 'choice'),
    "conv_model": (["resnet18", "resnet34", "squeezenet", "vgg16", "shufflenet"], 'choice'),
    "use_landmarks": ([False], 'choice'),
}
# resnet50, resnet18, resnet34, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, inception, googlenet, mobilenet, densenet, alexnet, vgg16, squeezenet, shufflenet, mnasnet

# max_trails for tuning is currently set to the number of combinations of the above hyperparameters
max_trails = 1
for key in tune_hp_ranges.keys():
    if tune_hp_ranges[key][1] == 'range':
        max_trails *= tune_hp_ranges[key][0][2]
    else:
        max_trails *= len(tune_hp_ranges[key][0])

TUNE_TARGET = "val_acc"
TUNE_MODE = "max"

EMOTION_INDEX = config.EMOTION_INDEX
EMOTION_INDEX_REVERSE = config.EMOTION_INDEX_REVERSE

softmax_len = config.softmax_len

############################################################################################################
# prediction defaults

SHOW_PRED_IMAGE = False  # This will display the predicted image in a new window
PREDICT_VERBOSE = True  # This will print the predicted emotion and other info on the console
GRAD_CAM = True  # Extract gradcam for the predicted image
GRAD_CAM_ON_VIDEO = False  # When video mode or webcam mode is on, this will display the gradcam overlay on the displayed face
############################################################################################################
# data filename label_schema:
# <dataset_name>_<s.no>_<emotion_index>.wav
