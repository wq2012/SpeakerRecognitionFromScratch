import librosa
import soundfile as sf
import os
import glob
import random
import torch
import numpy as np

import myconfig


def extract_features(flac_file):
    """Extract MFCC features from a flac file."""
    waveform, sample_rate = sf.read(flac_file)
    features = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=myconfig.N_MFCC)
    return features.transpose()


def get_spk_to_utts(data_dir):
    """Get the dict from speaker to list of utterances."""
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


def get_triplet_features_trimmed(spk_to_utts):
    """Get a triplet of trimmed anchor/pos/neg features."""
    anchor, pos, neg = get_triplet_features(spk_to_utts)
    while (anchor.shape[0] < myconfig.SEQ_LEN or
           pos.shape[0] < myconfig.SEQ_LEN or
           neg.shape[0] < myconfig.SEQ_LEN):
        anchor, pos, neg = get_triplet_features(spk_to_utts)
    return (trim_features(anchor),
            trim_features(pos),
            trim_features(neg))


def get_batched_triplet_input(spk_to_utts, batch_size):
    """Get batched triplet input for PyTorch."""
    input_arrays = []
    for _ in range(batch_size):
        anchor, pos, neg = get_triplet_features_trimmed(
            spk_to_utts)
        input_arrays += [anchor, pos, neg]
    batch_input = torch.from_numpy(np.stack(input_arrays)).float()
    return batch_input


def main():
    features = extract_features(os.path.join(
        myconfig.TEST_DATA_DIR, "61/70968/61-70968-0000.flac"))
    print("Shape of features:", features.shape)

    spk_to_utts = get_spk_to_utts(myconfig.TEST_DATA_DIR)
    triplet = get_triplet(spk_to_utts)
    triplet_features = get_triplet_features(spk_to_utts)
    print(triplet)
    print(triplet_features)


if __name__ == "__main__":
    main()
