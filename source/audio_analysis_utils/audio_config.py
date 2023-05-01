import source.config as config

# PERMANENT HPARAMS
# Model hyperparameters
initial_epoch = 50000
lr = 0.001
patience_epoch = 25  # This is used during early stopping
reduce_lr_factor = 0.5
reduce_lr_patience = 5

# Data hyperparameters
test_split_percentage = 0.1
READ_SAMPLE_RATE = 16000  # Consistent sr(num of samples per second) at which samples will be read
SIGNAL_SAMPLES_TO_CONSIDER = round(READ_SAMPLE_RATE * 0.1) # 1 sec. of audio is 22050 samples. 2 sec. is 44100 samples, etc. Signals below this length will be removed
CONSISTENT_SIGNAL_LENGTH = round(READ_SAMPLE_RATE * config.VIDEO_ANALYSE_WINDOW_SECS)  # 2.5 secs. Signals below this length will be padded, and signals above will be trimmed


# TUNING HPARAMS
# Model hyperparameters
# N_FFT ->  Interval we consider to apply FFT. Measured in # of samples
# HOP_LENGTH -> Sliding window for FFT. Measured in # of samples
# NUM_MFCC = 13 -> Number of coefficients to extract

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
    "dropout_rate": ([0.0, 0.3], 'choice'),
    "l1_l2_reg": ([0.0], 'choice'),
    "dense_units": ([128, 512], 'choice'),
    "num_layers": ([1, 3], 'choice'),
    "N_FFT": ([2048], 'choice'),
    "HOP_LENGTH": ([512], 'choice'),
    "NUM_MFCC": ([13], 'choice'),
    "layers_batch_norm": ([True], 'choice'),
    "conv_model": (["resnet18", "resnet101", "resnext50_32x4d"], 'choice'),
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
# data filename label_schema:
# <dataset_name>_<s.no>_<emotion_index>.wav
