import librosa
import soundfile as sf
import os

TRAIN_DATA_DIR = "/Users/quanw/Code/github/SpeakerRecognitionFromScratch/data/LibriSpeech/train-clean-100"
TEST_DATA_DIR = "/Users/quanw/Code/github/SpeakerRecognitionFromScratch/data/LibriSpeech/test-clean"

def extract_features(flac_file):
    waveform, sample_rate = sf.read(flac_file)
    features = librosa.feature.mfcc(waveform, sample_rate)
    return features

def main():
    features = extract_features(os.path.join(TRAIN_DATA_DIR, "19/198/19-198-0000.flac"))
    print(features.shape)


if __name__ == "__main__":
    main()