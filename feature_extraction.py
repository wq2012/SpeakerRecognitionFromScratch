import librosa
import soundfile as sf
import os
import glob
import random
import torch
import numpy as np
import csv

import myconfig


def extract_features(audio_file):
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = sf.read(audio_file)

    # Convert to mono-channel.
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform)

    # Convert to 16kHz.
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)

    features = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=myconfig.N_MFCC)
    return features.transpose()


def extract_sliding_windows(features):
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start: start + myconfig.SEQ_LEN, :])
        start += myconfig.SLIDING_WINDOW_STEP
    return sliding_windows


def get_librispeech_spk_to_utts(data_dir):
    """Get the dict from speaker to list of utterances for LibriSpeech."""
    flac_files = glob.glob(os.path.join(data_dir, "*", "*", "*.flac"))
    spk_to_utts = dict()
    for flac_file in flac_files:
        basename = os.path.basename(flac_file)
        split_name = basename.split("-")
        spk = split_name[0]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file]
        else:
            spk_to_utts[spk].append(flac_file)
    return spk_to_utts


def get_csv_spk_to_utts(csv_file):
    """Get the dict from speaker to list of utterances from CSV file."""
    spk_to_utts = dict()
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            spk = row[0].strip()
            utt = row[1].strip()
            if spk not in spk_to_utts:
                spk_to_utts[spk] = [utt]
            else:
                spk_to_utts[spk].append(utt)
    return spk_to_utts


def get_triplet(spk_to_utts):
    """Get a triplet of anchor/pos/neg samples."""
    pos_spk, neg_spk = random.sample(spk_to_utts.keys(), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]
    return (anchor_utt, pos_utt, neg_utt)


def get_triplet_features(spk_to_utts):
    """Get a triplet of anchor/pos/neg features."""
    anchor_utt, pos_utt, neg_utt = get_triplet(spk_to_utts)
    return (extract_features(anchor_utt),
            extract_features(pos_utt),
            extract_features(neg_utt))


def trim_features(features):
    """Trim features to SEQ_LEN."""
    full_length = features.shape[0]
    start = random.randint(0, full_length - myconfig.SEQ_LEN)
    return features[start: start + myconfig.SEQ_LEN, :]


class TrimmedTripletFeaturesFetcher:
    """The fetcher of trimmed features for multi-processing."""

    def __init__(self, spk_to_utts):
        self.spk_to_utts = spk_to_utts

    def __call__(self, _):
        """Get a triplet of trimmed anchor/pos/neg features."""
        anchor, pos, neg = get_triplet_features(self.spk_to_utts)
        while (anchor.shape[0] < myconfig.SEQ_LEN or
               pos.shape[0] < myconfig.SEQ_LEN or
               neg.shape[0] < myconfig.SEQ_LEN):
            anchor, pos, neg = get_triplet_features(self.spk_to_utts)
        return np.stack([trim_features(anchor),
                         trim_features(pos),
                         trim_features(neg)])


def get_batched_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batched triplet input for PyTorch."""
    fetcher = TrimmedTripletFeaturesFetcher(spk_to_utts)
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        input_arrays = pool.map(fetcher, range(batch_size))
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()
    return batch_input
