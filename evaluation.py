import torch

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
    pass
