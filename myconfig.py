# This file has the configurations of the experiments.
import os
import torch
import multiprocessing

# Paths of downloaded LibriSpeech datasets.
TRAIN_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "Code/github/SpeakerRecognitionFromScratch/data/LibriSpeech/train-clean-100")
TEST_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "Code/github/SpeakerRecognitionFromScratch/data/LibriSpeech/test-clean")

# Paths of CSV files where the first column is speaker, and the second column is
# utterance file.
# These will allow you to train/evaluate using other datasets than LibriSpeech.
# If given, TRAIN_DATA_DIR and/or TEST_DATA_DIR will be ignored.
TRAIN_DATA_CSV = ""
TEST_DATA_CSV = ""

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
BI_LSTM = True

# If false, use last frame of LSTM inference as aggregated output;
# if true, use mean frame of LSTM inference as aggregated output.
FRAME_AGGREGATION_MEAN = True

# If true, we use transformer instead of LSTM.
USE_TRANSFORMER = False

# Dimension of transformer layers.
TRANSFORMER_DIM = 32

# Number of encoder layers for transformer
TRANSFORMER_ENCODER_LAYERS = 2

# Number of heads in transformer layers.
TRANSFORMER_HEADS = 8

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

# Whether we are going to train with SpecAugment.
SPECAUG_TRAINING = False

# Parameters for SpecAugment training.
SPECAUG_FREQ_MASK_PROB = 0.3
SPECAUG_TIME_MASK_PROB = 0.3
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5

# Number of triplets to evaluate for computing Equal Error Rate (EER).
# Both the number of positive trials and number of negative trials will be
# equal to this number.
NUM_EVAL_TRIPLETS = 10000

# Step of threshold sweeping for computing Equal Error Rate (EER).
EVAL_THRESHOLD_STEP = 0.001

# Number of processes for multi-processing.
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

# Wehther to use GPU or CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
