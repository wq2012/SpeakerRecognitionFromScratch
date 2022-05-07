import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return y


def my_triplet(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA,
        torch.tensor(0))


def train_network(num_steps):
    spk_to_utts = feature_extraction.get_spk_to_utts(myconfig.TRAIN_DATA_DIR)
    losses = []

    encoder = SpeakerEncoder()

    # Train
    optimizer = optim.Adam(encoder.parameters(), lr=0.1)
    print('start training')
    for step in range(num_steps):
        optimizer.zero_grad()
        anchor, pos, neg = feature_extraction.get_triplet_features_trimmed(
            spk_to_utts)

        anchor_input = torch.from_numpy(anchor.transpose()).float()
        pos_input = torch.from_numpy(pos.transpose()).float()
        neg_input = torch.from_numpy(neg.transpose()).float()

        anchor_embedding = encoder(anchor_input)[-1, :]
        pos_embedding = encoder(pos_input)[-1, :]
        neg_embedding = encoder(neg_input)[-1, :]

        loss = my_triplet(
            anchor_embedding, pos_embedding, neg_embedding)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('step:', step, 'loss:', loss.item())
    print('finished training')
    return losses


def main():
    losses = train_network(1000)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    main()
