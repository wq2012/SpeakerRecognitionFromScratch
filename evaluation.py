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
    batch_input = torch.from_numpy(np.stack(sliding_windows)).float()
    batch_output = encoder(batch_input)[:, -1, :]
    aggregated_output = torch.mean(batch_output, dim=0, keepdim=False)
    return aggregated_output.data.numpy()
