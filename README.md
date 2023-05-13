### Summary:
This multimodal emotion detection model predicts a speaker's emotion using audio and image sequences from videos.
The repository contains two primary models: an audio tone recognition model with a CNN for audio-based emotion prediction, and a facial emotion recognition model using a CNN and optional mediapipe face landmarks for facial emotion prediction.
The third model combines a video clip's audio and image sequences, processed through an LSTM for speaker emotion prediction.
Hyperparameters such as landmark usage, CNN model selection, LSTM units, and dense layers are tuned for optimal accuracy using included modules.
For new datasets, follow the instructions below to retune the hyperparameters.
_______________
### Sample output for all 3 models:
#### Face model:

<table>
  <tr>
    <td><img src="display_files/child smile.png" width="300" style="margin-right: 10px;"></td>
    <td><img src="display_files/child_smile_emotion.jpg" width="200" style="margin-right: 10px;"></td>
    <td><img src="display_files/child_smile_grad_cam.jpg" width="200" style="margin-right: 10px;"></td>
  </tr>
  <tr>
    <td><img src="display_files/nervous_woman.png" width="300" style="margin-right: 10px;"></td>
    <td><img src="display_files/nervous_woman_emotion.jpg" width="200" style="margin-right: 10px;"></td>
    <td><img src="display_files/nervous_woman_grad_cam.jpg" width="200" style="margin-right: 10px;"></td>
  </tr>
  <tr>
    <td><img src="display_files/disgust_2.png" width="300" style="margin-right: 10px;"></td>
    <td><img src="display_files/disgust_emotion.jpg" width="200" style="margin-right: 10px;"></td>
    <td><img src="display_files/disgust_grad_cam.jpg" width="200" style="margin-right: 10px;"></td>
  </tr>
  <tr>
    <td><img src="display_files/angry.png" width="300" style="margin-right: 10px;"></td>
    <td><img src="display_files/angry_emotion.jpg" width="200" style="margin-right: 10px;"></td>
    <td><img src="display_files/angry_grad_cam.jpg" width="200"></td>
  </tr>
</table>

____________________
#### Audio model:

[angry_alex.mp4](display_files%2Fangry_alex.mp4)<br><br>
**_Prediction labels:_**<br>
Sad/Fear: 0.0%<br>
Neutral: 0.0%<br>
Happy: 0.0%<br>
Angry: 100.0%<br>
Surprise/Disgust: 0.0%<br>

[audio_happy.mp4](display_files%2Faudio_happy.mp4)<br><br>
**_Prediction labels:_**<br>
Sad/Fear: 0.0%<br>
Neutral: 0.0%<br>
Happy: 100.0%<br>
Angry: 0.0%<br>
Surprise/Disgust: 0.0%
_____________
#### Video/combined model:
<Program implemented, display items yet to be added>
_________________

### To run the program
1) Install python 3.10
2) Install everything you need
   - `git clone https://github.com/rishiswethan/Video-Audio-Face-Emotion-Recognition.git`
   - `cd Video-Audio-Face-Emotion-Recognition`
   - `git clone https://github.com/rishiswethan/pytorch_utils.git source/pytorch_utils`
   - `cd source/pytorch_utils && git checkout v1.0.3 && cd ../..`
   - `python -m venv venv`
   - Activate the virtual environment
     - Linux/MacOS: `source venv/bin/activate`
     - Windows: `venv\Scripts\activate`
   - `pip install -r requirements.txt`
   - `python -m spacy download en_core_web_lg`
   - To setup the GPU version of pytorch, follow the instructions in this [link](https://github.com/openai/whisper/discussions/47).
     A quick summary of the steps is given below:
       - `pip uninstall torch`
       - `pip cache purge`
       - `pip3 install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir`
         - There should be a download size of 2.3GB if it is downloading the GPU version correctly. If it's something like 200MB, that's the CPU version.
         - If that didn't work, simply install the CPU version for now. It'll slow down transcription but it'll work.
           - `pip install torch`
3) `python setup.py`
4) `python run.py`
   - When you run the program, you'll be asked to choose one of the 3 models. You can then follow the on-screen menu to run inference, training, etc.
   - Above instructions will set you up to run inference only. For things like training, preprocessing, etc, follow the instructions below.
_______________

### To preprocess data
1) Your data should be formatted like the example data [here](https://drive.google.com/file/d/1D1edFCrX6dffawxfzutSC5Fc5HXDITtS/view?usp=sharing). Have a look at `config.py` for a 
glimpse of the correct way to set file paths. The data should be in the following format. The preprocessing code will read the files names and infer the emotion from it.
    - Note that the data is not included in the repo. You'll have to download it yourself. `get_data.py` from all model folders will help you format this data. Have a look at the code and
   how the functions for each dataset are called. You can use the same functions to format your data, or write your own functions to get the below format.
    - Below are the list of datasets used. You can choose the ones you want to use from the `config.py` file. The default is the final commit.
      - **_Multimodal video or combined model_:**
        - OMG
        - CREMA-D
        - RAVDESS
        - SAVEE
      - **_Audio model_:**
        - CREMA-D
        - RAVDESS
        - TESS
        - SAVEE
      - **_Facial emotion detection model, that uses landmarks and image of cropped face_:**
        - FER
        - CK+
        - RAF

_______________
```html
   data 
    ├───training_AV                   // Audio visual data used for training the combined model
    │   ├───RAVDESS
    │   │   ├───train
    │   │   │  ├───RAVDESS_0_Neurtal.mp4                  // Example
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.mp4    // Correct format
    │   │   │  ├───...other videos
    │   │   ├───test
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.mp4
    │   // Other datasets must be added in the same format and added in the config.py file
    │    
    ├───training_faces                // Facial data used for training the landmarks and image based emotion detection model
    │   ├───FER
    │   │   ├───train
    │   │   │  ├───FER_0_Angry.png
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.png    // Correct format
    │   │   │  ├───...other images
    │   │   ├───test
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.png    // Test data from all faces datasets are used
    │            
    ├───extracted_audio               // Audio data used for training the audio model
    │   ├───RAVDESS
    │   │   ├───train
    │   │   │  ├───RAVDESS_0_Neurtal.wav                  // Example
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.wav    // Correct format
    │   │   │  ├───...other audio
    │   │   ├───test
    │   │   │  ├───<dataset_name>_<s.no>_<emotion>.wav    // We only use testing data from CREMA-D and TESS, this is just an example. The downloaded data will be empty here.
```

2) Choose the preprocess option from the run.py menu of your desired model.
3) You can choose the datasets you need to use from config.py file. Under the "All extracted" comment, you can choose the datasets you need to use.
The default is the final commit
4) You'll get many .npy files in the preprocessed_<model_name>_data folder. These are the preprocessed data files.
5) When you choose the train option from the run.py menu, these files will be used to train the model.
6) The SIMPLIFIED_EMOTIONS_MAP in the `config.py` file is used to simplify the emotions by combining them. Use it to combine emotions for which there is not enough data,
like "surprise" for which there was very little data, so I combined it with disgust. You can disable this simplification by setting the
`EMOTION_INDEX` to `FULL_EMOTION_INDEX`, and `EMOTION_INDEX_REVERSE` to `FULL_EMOTION_INDEX_REVERSE` in the `config.py` file.

___________
### To train the model
1) Choose the train option from the run.py menu
2) The tuned hyperparameters are in the best_hyperameters.json file that's in the respective model's folder. You can change them if you want, but the default values are the result of my hyperparameter tuning.
3) To tune them yourself, you can set the various ranges in the <model_name>_config.py files of each model. You can always reset them back to the default, so
should you wish to try tuning them with your own ranges, feel free to do so.
4) The trained model will be saved the 'models' folder.

__________
### To use new datasets
1) You'd have to write code to convert the labels of the new dataset to the format of the labels of the datasets used here. I've mentioned the correct format below
   * `<dataset_name>_<s.no>_<emotion_index>.<file_extension>`
   * emotion index is the index of the emotion in the `FULL_EMOTION_INDEX` in the `config.py` file. For example, if the emotion is 'happy', the index is 3.
2) Make sure emotion index is in the same order as the `FULL_EMOTION_INDEX` dict of the `config.py` file. Simplifying is only done when the preprocessed data
is used for training. Before that, we only use all the emotions in the `FULL_EMOTION_INDEX` dict as you can see in the dataset provided.
3) If you are using new videos or audio, you'll have to make sure each video and audio file is of similar length, as in the `config.py`
file's `VIDEO_ANALYSE_WINDOW_SECS` variable. If videos in the training set are longer, only the last x seconds will be used. If they are shorter, they will be padded to meet the window size.
   * Example using the default 2.5 secs window size:
       * 3.5 sec video/audio, last 2.5 secs will be used
       * 1.5 sec video/audio will be padded to 2.5 secs
4) If your video/audio is significantly longer than the above length, I recommend you split the video/audio into the above length and label them accordingly.
I did this with the OMG and MELD dataset, which originally, only had timestamps with its corresponding emotion.
I already have the code to split data in the `get_data.py` file of the combined model. You can use it as well.
5) You'll have to include this correctly labelled dataset path in the corresponding _`ALL_EXTRACTED_<model_name>_FOLDERS`_ path list. You'll have a clearer understanding if you look at how
I wrote the various paths for this in the config.py file.

____________
### How everything works:
* #### General summary:
    - ##### Programs:
      - `run.py` is the main file. It contains the menu to run the various options. You can use this as a reference to see how to call the various functions in your
  customised implementation.
      - `setup.py` nothing fancy, just creates the empty folders like inputs, outputs, etc.
      - `source/config.py` contains all the file paths, and general constants used by all the models.
      - `source/<model_name>/<model_name>_config.py` contains all the model specific constants, such as initial learning rate, scheduler configs and hyperparameters tuning ranges.
      - `source/<model_name>/<model_name>_model.py` contains the model architecture.
      - `source/<model_name>/<model_name>_preprocess_main.py` contains the code to preprocess the data and store it as .npy files in the 
  `data/preprocessed_<model_name>_data` folder
      - `source/<model_name>/utils.py` contains all the utility functions used by the model. Some of them are used by all the models, and some are specific to the model.
      - `source/<model_name>/predict.py` contains the code to predict the emotion of a file.
      - `source/<model_name>/get_data.py` contains the code to format the data into the correct format for the preprocessing code to read, as mentioned above.

    - #### Folders:
      - `data/preprocessed_<model_name>_data` contains the preprocessed `.npy` data files. These are the files that are used to train the model. 
      - `data/original_<dataset_name>_data` contains the original, unaltered data files. You can format them into the `training_AV, training_face and extracted audio`
      folders using the `get_data.py` program, or your own.
      - `models` contains the trained models, best hyperparameters and the tuning logs. Note that the model trained based on the best config, may not be have the exact
      performance as you see in the tuning logs. This is because, the model is retrained using random initialization, and it won't converge to the same minima as it did during tuning.
      But the difference in performance should be negligible.
      - `input_files` contains the input files of the predict program. This is where the file name of your input will be searched for, when you run `run.py`.
      - `output_files` contains the output files of the predict program.

    - ##### Working of the program:
      * We first train the audio tone model and image/landmark models separately. We then use the trained models to train the combined model, which uses both audio and image data,
        and uses softmax emotions from the transcript extracted from the audio, to run sentiment analysis. Transcripts for both training and prediction are generated using whisper.
      * For all models, we preprocess the data and store it as .npy files in the `data/preprocessed_<model_name>_data` folder. Preprocessing takes way too long, and it's not
        possible to preprocess the data on the fly, so we preprocess it once and store it. This also makes it easier to tune the hyperparameters, as we can use the same preprocessed
        data for all the hyperparameter tuning runs.

* #### Audio model:
    - MFCC features are extracted from the audio file.
    - The features are then passed through a CNN model.
    - The output of the CNN model is then passed through dense layers to get the emotion prediction. This model only uses the tone of the audio to make predictions.
* #### Image model:
    - The image is passed through a CNN model.
    - Landmarks are extracted from the image.
    - The CNN output and the landmarks are concatenated and passed through dense layers to get the emotion prediction.
* #### Combined model(video):
    - The video is cut into `VIDEO_ANALYSE_WINDOW_SECS` second videos.
    - Audio is extracted from each of these videos.
    - The audio is passed through the audio model.
    - Audio is passed through the transcript and sentiment/emotion analysis model to get the sentiment and emotion of the transcript of each video.
    - FPS of the video is reduced to `FRAME_RATE` frames per second.
    - Each frame is passed through the image model.
    - The image model outputs are concatenated and arranged as a time series and passed through LSTM layer(s) so that temporal information through frames is captured.
    - The frames time series result, audio CNN result and sentiment results are passed through dense layer(s) to get the final emotion prediction.

_____________
### Note:
* The program uses pytorch and uses a GPU by default, if it's found, else the CPU is used. Make sure you have the GPU setup correctly if you want to run it on a GPU.
Running the program to predict shouldn't take long in CPUs, but training will take too long. I recommend using a GPU for training.
* All tuning, training and preprocessing was done on a
NVIDIA GeForce RTX 4090 GPU, with RAM of 64GB.
* You'll get a memory error if you have too little RAM when training or preprocessing. Around 40GB should be sufficient for preprocessing. You'll need
lesser RAM for training and less than 8GB can run inference very easily. If you want to reduce the RAM usage,
  * _When preprocessing_, you can store the data in the .npy files or other ways more frequently, instead of all at once as I did. It shouldn't really reduce the preprocessing time by much. This also
  allows you to use the DataLoader class that is already overridden in all models, to load the preprocessed batches of numpy arrays from the disk, instead of loading all the data into memory.
  * _When training_, you can use a custom DataGenerator using which, you can feed each batch from the disk,
  instead of loading all the data into memory. This will reduce the RAM usage, but will increase the training time as the data will be loaded from the disk each time and inference will have to be run for
  models like whisper, mediapipe, etc.
* The program was developed and tested on python 3.10 It should work on python 3.9 and above, but I haven't tested it on those versions.* 

_____________
### Contributing:
I've made this project public so that it can be used by anyone who wants to use it. I've tried to make it as easy to use as possible, but if you have any questions or suggestions, create an issue or discuss it in the discussions section.
I'll try to answer them as soon as possible. If you want to contribute, create a pull request. I'll try to review it as soon as possible.
