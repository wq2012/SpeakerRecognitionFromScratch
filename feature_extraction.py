import librosa
import soundfile as sf
import random
import torch
import numpy as np
import functools

import myconfig
import dataset
import specaug

SAMPLE_RATE = 16000


def extract_features(audio_file):
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = sf.read(audio_file)

    # Convert to mono-channel.
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Convert to 16kHz.
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, sample_rate, SAMPLE_RATE)

    features = librosa.feature.mfcc(
        y=waveform, sr=SAMPLE_RATE, n_mfcc=myconfig.N_MFCC)
    return features.transpose()


def extract_sliding_windows(features):
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start: start + myconfig.SEQ_LEN, :])
        start += myconfig.SLIDING_WINDOW_STEP
    return sliding_windows


def get_triplet_features(spk_to_utts):
    """Get a triplet of anchor/pos/neg features."""
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    return (extract_features(anchor_utt),
            extract_features(pos_utt),
            extract_features(neg_utt))


def trim_features(features, apply_specaug):
    """Trim features to SEQ_LEN."""
    full_length = features.shape[0]
    start = random.randint(0, full_length - myconfig.SEQ_LEN)
    trimmed_features = features[start: start + myconfig.SEQ_LEN, :]
    if apply_specaug:
        trimmed_features = specaug.apply_specaug(trimmed_features)
    return trimmed_features


def get_trimmed_triple_features(_, spk_to_utts):
    """Get a triplet of trimmed anchor/pos/neg features."""
    anchor, pos, neg = get_triplet_features(spk_to_utts)
    while (anchor.shape[0] < myconfig.SEQ_LEN or
            pos.shape[0] < myconfig.SEQ_LEN or
            neg.shape[0] < myconfig.SEQ_LEN):
        anchor, pos, neg = get_triplet_features(spk_to_utts)
    return np.stack([trim_features(anchor, myconfig.SPECAUG_TRAINING),
                        trim_features(pos, myconfig.SPECAUG_TRAINING),
                        trim_features(neg, myconfig.SPECAUG_TRAINING)])


def get_batched_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batched triplet input for PyTorch."""
    fetcher = functools.partial(get_trimmed_triple_features,
                                spk_to_utts=spk_to_utts)
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        input_arrays = pool.map(fetcher, range(batch_size))
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()
    return batch_input
