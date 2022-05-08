import torch
import numpy as np

import feature_extraction
import neural_net
import myconfig


def load_saved_model(saved_model):
    """Load the saved model."""
    encoder = neural_net.SpeakerEncoder()
    var_dict = torch.load(saved_model)
    encoder.load_state_dict(var_dict["encoder_state_dict"])
    return encoder


def run_inference(features, encoder):
    """Get the embedding of an utterance using the encoder."""
    sliding_windows = feature_extraction.extract_sliding_windows(features)
    if not sliding_windows:
        return None
    batch_input = torch.from_numpy(np.stack(sliding_windows)).float()
    batch_output = encoder(batch_input)[:, -1, :]
    aggregated_output = torch.mean(batch_output, dim=0, keepdim=False)
    return aggregated_output.data.numpy()


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_scores(encoder, num_eval_triplets=myconfig.NUM_EVAL_TRIPLETS):
    """Compute cosine similarity scores from testing data."""
    labels = []
    scores = []
    spk_to_utts = feature_extraction.get_spk_to_utts(myconfig.TEST_DATA_DIR)
    triplets_evaluated = 0
    while triplets_evaluated < num_eval_triplets:
        anchor, pos, neg = feature_extraction.get_triplet_features(spk_to_utts)
        anchor_embedding = run_inference(anchor, encoder)
        pos_embedding = run_inference(pos, encoder)
        neg_embedding = run_inference(neg, encoder)
        if ((anchor_embedding is None) or
            (pos_embedding is None) or
                (neg_embedding is None)):
            # Some utterances might be smaller than a single sliding window.
            continue
        labels.append(1)
        scores.append(cosine_similarity(anchor_embedding, pos_embedding))
        labels.append(0)
        scores.append(cosine_similarity(anchor_embedding, neg_embedding))
        triplets_evaluated += 1
    return (labels, scores)


def compute_eer(labels, scores):
    """Compute the Equal Error Rate (EER)."""
    assert len(labels) == len(scores)
    eer_threshold = None
    eer = None
    min_delta = 1
    threshold = 0.0
    while threshold < 1.0:
        accept = [score >= threshold for score in scores]
        fa = [a and (1-l) for a, l in zip(accept, labels)]
        fr = [(1-a) and l for a, l in zip(accept, labels)]
        far = sum(fa) / (len(labels) - sum(labels))
        frr = sum(fr) / sum(labels)
        delta = abs(far - frr)
        if delta < min_delta:
            min_delta = delta
            eer = (far + frr) / 2
            eer_threshold = threshold
        threshold += myconfig.EVAL_THRESHOLD_STEP

    return eer, eer_threshold


def run_eval():
    """Run evaluation of the saved model on test data."""
    encoder = load_saved_model(myconfig.SAVED_MODEL_PATH)
    labels, scores = compute_scores(encoder, myconfig.NUM_EVAL_TRIPLETS)
    eer, eer_threshold = compute_eer(labels, scores)
    print("eer_threshold =", eer_threshold, "eer =", eer)


if __name__ == "__main__":
    run_eval()
