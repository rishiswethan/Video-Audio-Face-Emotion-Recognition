import librosa
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import time
import nltk
from nltk.stem import WordNetLemmatizer
import re
import spacy

import source.whisper.whisper as whisper
import source.config as config

import source.audio_analysis_utils.utils as audio_utils

tokenizer = None
sentiment_classifier = None
emotion_classifier = None

models = {}
models_in_use = []


# Load models
def init(max_threads=1):
    global models, models_in_use, tokenizer, sentiment_classifier, emotion_classifier

    print("Loading transcription and sentiment models...")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_classifier = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                          return_all_scores=True)

    for i in range(max_threads):
        transcribe_model = whisper.load_model("medium.en")
        models[i] = transcribe_model
    print("Transcription and sentiment models loaded")


def delete_models():
    global models, sentiment_classifier, emotion_classifier, tokenizer, models_in_use

    tokenizer = None
    sentiment_classifier = None
    emotion_classifier = None

    models = {}
    models_in_use = []


# Transcribe speech using whisper
def transcribe_audio(file_path=None, audio=None):
    global models_in_use, models

    while len(models_in_use) >= len(models):
        time.sleep(0.1)

    free_models = list(set(models.keys()) - set(models_in_use))
    transcribe_model = models[free_models[0]]
    models_in_use.append(free_models[0])

    if audio is not None:
        result = transcribe_model.transcribe(audio, condition_on_previous_text=False)
    else:
        result = transcribe_model.transcribe(file_path, condition_on_previous_text=False)

    models_in_use.remove(free_models[0])

    return result["text"]


# Sentiment and emotion analysis
def get_text_sentiment(transcription_string):
    global sentiment_classifier, tokenizer

    def convert_logits_to_softmax(logits):
        softmax = torch.nn.Softmax(dim=1)
        softmax_list = softmax(logits).tolist()[0]
        return softmax_list

    # Get sentiment
    tokenizer_output = tokenizer(transcription_string, return_tensors="pt")
    returned_sentiment = sentiment_classifier(**tokenizer_output)

    # Get emotion
    returned_emotion = emotion_classifier(transcription_string)

    # Softmax
    organised_sentiments = convert_logits_to_softmax(returned_sentiment.logits)

    # emotion_label: score and softmax
    # anger disgust fear joy neutral sadness surprise
    organised_emotions = {}
    emotions_softmax = []
    for emotion in returned_emotion[0]:
        organised_emotions[emotion["label"]] = emotion["score"]
        emotions_softmax.append(emotion["score"])

    highest_emotion_score = max(organised_emotions.values())
    highest_emotion_label = [k for k, v in organised_emotions.items() if v == highest_emotion_score][0]

    max_sentiment = max(organised_sentiments)
    max_sentiment_index = organised_sentiments.index(max_sentiment)
    pos_or_neg = "Positive" if max_sentiment_index == 1 else "Negative"

    emotion_sentiment_softmax = emotions_softmax + organised_sentiments

    returned_sentiment = {
        "max_sentiment_label": pos_or_neg,
        "max_sentiment_value": max_sentiment,
        "sentiment_softmax": organised_sentiments,
        "max_emotion_label": highest_emotion_label,
        "max_emotion_value": highest_emotion_score,
        "emotion_softmax": organised_emotions,
        "emotion_sentiment_softmax": emotion_sentiment_softmax
    }

    return returned_sentiment


# Get list of good words
def _get_list_of_wanted_words(string, print_flag=False):
    # Download this if you haven't already.
    # python -m spacy download en_core_web_lg
    nlp = spacy.load('en_core_web_lg')

    # excluded tags
    excluded_pos = ["ADV", "ADP", "AUX", "ADJ", "CCONJ", "CONJ", "DET", "INTJ", "NUM", "PART", "PRON", "NUM", "PATH", "PUNCT", "X", "SYM", "SCONJ", "SPACE"]
    # excluded detailed tags
    excluded_detailed_tags = {"VERB": ["BES", "HVS", "MD", "VB", "VBD", "VBP", "VBZ"]}

    word_correct_or_not = {}
    wanted_words = []
    for nlp_word in nlp(string):
        # Find main tags
        if nlp_word.pos_ in excluded_pos:
            word_correct_or_not[nlp_word] = False
        # Find excluded detailed tags
        elif nlp_word.pos_ in excluded_detailed_tags:
            if nlp_word.tag_ in excluded_detailed_tags[nlp_word.pos_]:
                word_correct_or_not[nlp_word] = False
        # Find if word only contains letters
        elif re.search('[^a-zA-Z]', nlp_word.text):
            word_correct_or_not[nlp_word] = False
        else:
            word_correct_or_not[nlp_word] = True
            wanted_words.append(str(nlp_word))

        if print_flag:
            print(nlp_word.pos_, end=" ")
            print(nlp_word.tag_)
            print(nlp_word, word_correct_or_not[nlp_word])
            print("......")
    return wanted_words


# Find most common words
def find_common_words(transcription_string):
    nltk.download('wordnet')

    # Get list of good words
    wanted_words = _get_list_of_wanted_words(transcription_string)

    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in wanted_words:
        word = word.lower()
        # Lemmatize word
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemmatized_word)

    # Find most common words
    word_count = {}
    for word in lemmatized_words:
        # print(word)
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count


# Tests
if __name__ == "__main__":
    init()

    # -- Example of transcribing audio
    # transcription = transcribe_audio(config.INPUT_FOLDER_PATH + "joe_rogan_alex.mp4")
    # print(transcription)
    # print(get_text_sentiment(transcription[0]))

    # -- Example of transcription directly from audio. Use this if you get FileNotFoundError when passing filename directly to transcribe_audio()
    y, sr = librosa.load(config.INPUT_FOLDER_PATH + "joe_rogan_alex.mp4")
    transcription = transcribe_audio(audio=y)
    print(transcription)

    # -- Example of finding emotion and sentiment
    # returned_sentiment = get_text_sentiment("A woman is friendly and I took it the wrong way. why do I do this to myself")
    # print(returned_sentiment)
    # returned_sentiment = get_text_sentiment("I am happy")
    # print(returned_sentiment)

    # -- Example of finding common words and getting list of unwanted words
    # find_common_words("There are many rocks. I love throwing rocks at people")
    # print(_get_list_of_wanted_words("Function words serve grammatical functions in sentences such as linking, connecting, or indicating tense, possession, etc. They have less lexical meaning than content words and are usually shorter and more common in language."))

    # -- Example of finding common words from text file
    # with open(config.INPUT_FOLDER_PATH + "test.txt", "r") as f:
    #     text = f.read().replace("\n", " ")
    #     print(text)
    #     print(find_common_words(text))

    # -- Example of finding common words from video after transcribing it
    # text = transcribe_audio(config.INPUT_FOLDER_PATH + "How My Parents Fight.mp4")
    # print(text)
    # print(find_common_words(text))

    delete_models()