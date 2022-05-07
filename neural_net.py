import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import feature_extraction
import myconfig


class SpeakerEncoder(nn.Module):

    def __init__(self):
        super(SpeakerEncoder, self).__init__()
        # Define the RNN layer and a final linear layer
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(
            myconfig.LSTM_NUM_LAYERS, x.shape[0],  myconfig.LSTM_HIDDEN_SIZE)
        c0 = torch.zeros(
            myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE)
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return y


def my_triplet(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA,
        torch.tensor(0.0))


def train_network(num_steps, saved_model):
    start_time = time.time()
    losses = []
    spk_to_utts = feature_extraction.get_spk_to_utts(myconfig.TRAIN_DATA_DIR)

    encoder = SpeakerEncoder()

    # Train
    optimizer = optim.Adam(encoder.parameters(), lr=0.1)
    print("Start training")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Build batched input.
        input_arrays = []
        for _ in range(myconfig.BATCH_SIZE):
            anchor, pos, neg = feature_extraction.get_triplet_features_trimmed(
                spk_to_utts)
            input_arrays += [anchor.transpose(), pos.transpose(),
                             neg.transpose()]
        batch_input = torch.from_numpy(np.stack(input_arrays)).float()

        # Compute loss.
        batch_output = encoder(batch_input)[:, -1, :]
        loss = torch.tensor(0.0)
        for batch in range(myconfig.BATCH_SIZE):
            loss += my_triplet(
                batch_output[batch * 3, :],
                batch_output[batch * 3 + 1, :],
                batch_output[batch * 3 + 2, :])

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "loss:", loss.item())

    training_time = time.time() - start_time
    print("finished training in", training_time, "seconds")
    torch.save({"encoder_state_dict": encoder.state_dict(),
                "losses": losses,
                "training_time": training_time},
               saved_model)
    return losses


def main():
    losses = train_network(10, myconfig.SAVED_MODEL_PATH)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    main()
