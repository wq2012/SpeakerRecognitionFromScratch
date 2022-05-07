import torch
import torch.nn as nn
import torch.optim as optim
import feature_extraction
import myconfig


class SpeakerEncoder(nn.Module):

    def __init__(self):
        super(SpeakerEncoder, self).__init__()
        # Define the RNN layer and a final linear layer
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS)

    def forward(self, x):
        h0 = torch.zeros(myconfig.LSTM_NUM_LAYERS, myconfig.LSTM_HIDDEN_SIZE)
        c0 = torch.zeros(myconfig.LSTM_NUM_LAYERS, myconfig.LSTM_HIDDEN_SIZE)
        y, hn, cn = self.lstm(x, (h0, c0))
        return y


def my_triplet(anchor, pos, neg):
    """Define triplet loss here."""
    return torch.maximum(
        nn.CosineSimilarity(anchor, neg) -
        nn.CosineSimilarity(anchor, pos) + myconfig.TRIPLET_ALPHA,
        0)
