import os

# Path to the directory containing the audio dataset.
PATH_TO_DATASET = os.path.join(
    os.path.expanduser("~"), "Downloads/CN-Celeb_flac/data")

# Extention of the audio files.
AUDIO_FORMAT = ".flac"

# After we split the full audio path, the index indicating which part is
# the speaker label.
SPEAKER_LABEL_INDEX = -2

# Path to the output CSV file.
OUTPUT_CSV = "CN-Celeb.csv"


if __name__ == "__main__":
    # Find all files in PATH_TO_DATASET with the extenion of AUDIO_FORMAT.
    all_files = [os.path.join(dirpath, filename)
                 for dirpath, _, files in os.walk(PATH_TO_DATASET)
                 for filename in files if filename.endswith(AUDIO_FORMAT)]

    # Prepare CSV text content.
    content = []
    for filename in all_files:
        speaker = filename.split(os.sep)[SPEAKER_LABEL_INDEX]
        content.append(",".join([speaker, filename]))

    # Write CSV.
    with open(OUTPUT_CSV, "w") as f:
        f.write("\n".join(content))
