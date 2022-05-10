import numpy as np
import random

import myconfig

FREQ_MASK_PROB = 0.2
TIME_MASK_PROB = 0.2

FREQ_MASK_MAX_WIDTH = myconfig.N_MFCC // 10
TIME_MASK_MAX_WIDTH = myconfig.SEQ_LEN // 10


def apply_specaug(features):
    """Apply SpecAugment to features."""
    seq_len, n_mfcc = features.shape
    outputs = features
    mean_feature = np.mean(features)

    # Frequancy masking.
    if random.random() < FREQ_MASK_PROB:
        width = random.randint(1, FREQ_MASK_MAX_WIDTH)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feature

    # Time masking.
    if random.random() < TIME_MASK_PROB:
        width = random.randint(1, TIME_MASK_MAX_WIDTH)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feature

    return outputs
