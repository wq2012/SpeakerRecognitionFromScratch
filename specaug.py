import numpy as np
import random

import myconfig


def apply_specaug(features):
    """Apply SpecAugment to features."""
    seq_len, n_mfcc = features.shape
    outputs = features
    mean_feature = np.mean(features)

    # Frequancy masking.
    if random.random() < myconfig.SPECAUG_FREQ_MASK_PROB:
        width = random.randint(1, myconfig.SPECAUG_FREQ_MASK_MAX_WIDTH)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feature

    # Time masking.
    if random.random() < myconfig.SPECAUG_TIME_MASK_PROB:
        width = random.randint(1, myconfig.SPECAUG_TIME_MASK_MAX_WIDTH)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feature

    return outputs
