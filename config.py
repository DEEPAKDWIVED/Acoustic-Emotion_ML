"""Configuration file for emotion recognition project."""

import os

# Dataset paths
DATASET_RAVDESS = r"C:\Emotion DataSet\RAVDESS\Audio_Speech_Actors_01-24"
DATASET_AUDIODATA = r"C:\Emotion DataSet\AudioData"
DATASET_TESS = r"C:\Emotion DataSet\TESS"
OUTPUT_AUDIO_DIR = r"C:\Emotion DataSet\Extracted_Audio_With_Noise"

# Output paths
OUTPUT_CSV_ORIGINAL = "../data/combined_emotion_features_with_noise.csv"
OUTPUT_CSV_BALANCED = "../data/balanced_emotion_features.csv"

# Model paths
MODEL_SAVE_DIR = "../models"

# Emotion label mappings
RAVDESS_EMOTION_LABELS = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

AUDIODATA_EMOTION_LABELS = {
    "a": "Angry", "d": "Disgust", "f": "Fearful", "h": "Happy",
    "n": "Neutral", "sa": "Sad", "su": "Surprised"
}

TESS_EMOTION_LABELS = {
    "OAF_angry": "Angry", "YAF_angry": "Angry",
    "OAF_disgust": "Disgust", "YAF_disgust": "Disgust",
    "OAF_Fear": "Fearful", "YAF_Fear": "Fearful",
    "OAF_happy": "Happy", "YAF_happy": "Happy",
    "OAF_neutral": "Neutral", "YAF_neutral": "Neutral",
    "OAF_Pleasant_surprise": "Surprised", "YAF_Pleasant_surprise": "Surprised",
    "OAF_Sad": "Sad", "YAF_Sad": "Sad"
}

# Mapping primary emotions to subsets
EMOTION_SUBSET_MAPPING = {
    "Sad": "Negative", "Angry": "Negative", "Fearful": "Negative", "Disgust": "Negative",
    "Happy": "Positive", "Surprised": "Positive",
    "Calm": "Neutral", "Neutral": "Neutral"
}

# Feature extraction parameters
N_MFCC = 40
NOISE_DECIMAL = 0.1

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_TEST_SPLIT = 0.50

# Disable numba to avoid multiprocessing issues
os.environ["NUMBA_DISABLE_JIT"] = "1"
