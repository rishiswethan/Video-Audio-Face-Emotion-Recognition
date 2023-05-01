import source.config as config


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
    # Training hyperparameters
    "batch_size": ([128], 'choice'),
    "dropout_rate": ([0.0, 0.5], 'choice'),
    "l1_l2_reg": ([1e-06], 'choice'),
    "dense_units": ([128, 512], 'choice'),
    "num_layers": ([2], 'choice'),
    "sequence_model_dense_units": ([128, 512], 'choice'),
    "sequence_model_layers": ([1, 3], 'choice'),
    "layers_batch_norm": ([True], 'choice'),
    # Preprocessing hyperparameters
    "normalise": ([True], 'choice'),
    # Audio hyperparameters
    "N_FFT": ([2048], 'choice'),
    "HOP_LENGTH": ([512], 'choice'),
    "NUM_MFCC": ([13], 'choice'),
    "prev_layers_trainable": ([True], 'choice'),
}

# max_trails for tuning is currently set to the number of combinations of the above hyperparameters
max_trails = 1
for key in tune_hp_ranges.keys():
    if tune_hp_ranges[key][1] == 'range':
        max_trails *= tune_hp_ranges[key][0][2]
    else:
        max_trails *= len(tune_hp_ranges[key][0])

VIDEO_ANALYSE_WINDOW_SECS = config.VIDEO_ANALYSE_WINDOW_SECS  # This is the window size for video analysis. Videos will be cropped to chunks of this size
FRAME_RATE = config.FRAME_RATE  # This is the frame rate of the video

test_split_percentage = 0.15

AUDIO_INPUT_SHAPE = config.AUDIO_INPUT_SHAPE
TEXT_SENTIMENT_INPUT_SHAPE = config.TEXT_SENTIMENT_INPUT_SHAPE

lr = 0.001

reduce_lr_factor = 0.5  # This is used for learning rate scheduler. The learning rate will be reduced by this factor
reduce_lr_patience = 5  # This is used for learning rate scheduler. The learning rate will be reduced if the validation loss does not improve for this many epochs
early_stopping_patience_epoch = 25  # This is used for early stopping. The training will be stopped if the validation loss does not improve for this many epochs
softmax_len = config.softmax_len

initial_epoch = 50000

MAX_PREPROCESS_THREADS = 5  # This is the number of threads used for preprocessing. Reduce this if you are getting memory errors or CUDA errors
MAX_TRAINING_THREADS = 2  # This is the number of threads used for training. Reduce this if you are getting memory errors or CUDA errors

DISABLE_SENTIMENT = True  # Set this to True if you want to disable sentiment analysis

TUNE_TARGET = "val_acc"
TUNE_MODE = "max"
