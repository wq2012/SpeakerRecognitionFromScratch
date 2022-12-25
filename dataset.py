import random
import os
import glob
import csv


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
    pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    # Retry if too few positive utterances.
    while len(spk_to_utts[pos_spk]) < 2:
        pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]
    return (anchor_utt, pos_utt, neg_utt)
