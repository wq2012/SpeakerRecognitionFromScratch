# This file has the configurations of the experiments.
import os
import torch
import multiprocessing

# Path of downloaded LibriSpeech datasets.
TRAIN_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "Code/github/SpeakerRecognitionFromScratch/data/LibriSpeech/train-clean-100")
TEST_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "Code/github/SpeakerRecognitionFromScratch/data/LibriSpeech/test-clean")

# Path of save model.
SAVED_MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    "Code/github/SpeakerRecognitionFromScratch/saved_model/saved_model.pt")

# Number of MFCCs for librosa.feature.mfcc.
N_MFCC = 40

# Hidden size of LSTM layers.
LSTM_HIDDEN_SIZE = 64

# Number of LSTM layers.
LSTM_NUM_LAYERS = 3

# Whether to use bi-directional LSTM.
BI_LSTM = False

# If false, use last frame of LSTM inference as aggregated output;
# if true, use mean frame of LSTM inference as aggregated output.
FRAME_AGGREGATION_MEAN = False

# Sequence length of the sliding window for LSTM.
SEQ_LEN = 100  # 3.2 seconds

# Sliding window step for LSTM inference.
SLIDING_WINDOW_STEP = 50  # 1.6 seconds

# Alpha for the triplet loss.
TRIPLET_ALPHA = 0.1

# How many triplets do we train in a single batch.
BATCH_SIZE = 8

# Learning rate.
LEARNING_RATE = 0.0001

# Save a model to disk every these many steps.
SAVE_MODEL_FREQUENCY = 10000

# Number of steps to train.
TRAINING_STEPS = 100000

# Number of triplets to evaluate for computing Equal Error Rate (EER).
# Both the number of positive trials and number of negative trials will be
# equal to this number.
NUM_EVAL_TRIPLETS = 1000

# Step of threshold sweeping for computing Equal Error Rate (EER).
EVAL_THRESHOLD_STEP = 0.001

# Number of processes for multi-processing.
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

# Wehther to use GPU or CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
