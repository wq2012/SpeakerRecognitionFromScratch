import os
import torch
import unittest
import numpy as np

import feature_extraction
import neural_net
import evaluation
import myconfig


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.spk_to_utts = feature_extraction.get_spk_to_utts(
            myconfig.TEST_DATA_DIR)

    def test_extract_features(self):
        features = feature_extraction.extract_features(os.path.join(
            myconfig.TEST_DATA_DIR, "61/70968/61-70968-0000.flac"))
        self.assertEqual(features.shape, (154, myconfig.N_MFCC))

    def test_extract_sliding_windows(self):
        features = feature_extraction.extract_features(os.path.join(
            myconfig.TEST_DATA_DIR, "61/70968/61-70968-0000.flac"))
        sliding_windows = feature_extraction.extract_sliding_windows(features)
        self.assertEqual(len(sliding_windows), 2)
        self.assertEqual(sliding_windows[0].shape,
                         (myconfig.SEQ_LEN, myconfig.N_MFCC))

    def test_get_spk_to_utts(self):
        self.assertEqual(len(self.spk_to_utts.keys()), myconfig.N_MFCC)
        self.assertEqual(len(self.spk_to_utts["121"]), 62)

    def test_get_triplet(self):
        anchor1, pos1, neg1 = feature_extraction.get_triplet(self.spk_to_utts)
        anchor1_spk = os.path.basename(anchor1).split("-")[0]
        pos1_spk = os.path.basename(pos1).split("-")[0]
        neg1_spk = os.path.basename(neg1).split("-")[0]
        self.assertEqual(anchor1_spk, pos1_spk)
        self.assertNotEqual(anchor1_spk, neg1_spk)

        anchor2, pos2, neg2 = feature_extraction.get_triplet(self.spk_to_utts)
        anchor2_spk = os.path.basename(anchor2).split("-")[0]
        pos2_spk = os.path.basename(pos2).split("-")[0]
        neg2_spk = os.path.basename(neg2).split("-")[0]
        self.assertNotEqual(anchor1_spk, anchor2_spk)
        self.assertNotEqual(pos1_spk, pos2_spk)
        self.assertNotEqual(neg1_spk, neg2_spk)

    def test_get_triplet_features(self):
        anchor, pos, neg = feature_extraction.get_triplet_features(
            self.spk_to_utts)
        self.assertEqual(myconfig.N_MFCC, anchor.shape[1])
        self.assertEqual(myconfig.N_MFCC, pos.shape[1])
        self.assertEqual(myconfig.N_MFCC, neg.shape[1])

    def test_get_triplet_features_trimmed(self):
        anchor, pos, neg = feature_extraction.get_triplet_features_trimmed(
            self.spk_to_utts)
        self.assertEqual(anchor.shape, (myconfig.SEQ_LEN, myconfig.N_MFCC))
        self.assertEqual(pos.shape, (myconfig.SEQ_LEN, myconfig.N_MFCC))
        self.assertEqual(neg.shape, (myconfig.SEQ_LEN, myconfig.N_MFCC))

    def test_get_batched_triplet_input(self):
        batch_input = feature_extraction.get_batched_triplet_input(
            self.spk_to_utts, batch_size=4)
        self.assertEqual(batch_input.shape, torch.Size(
            [3 * 4, myconfig.SEQ_LEN, myconfig.N_MFCC]))


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.encoder = neural_net.SpeakerEncoder()

    def test_get_triplet_loss1(self):
        anchor = torch.tensor([0.0, 1.0])
        pos = torch.tensor([0.0, 1.0])
        neg = torch.tensor([0.0, 1.0])
        loss = neural_net.get_triplet_loss(anchor, pos, neg)
        loss_value = loss.data.numpy().item()
        self.assertAlmostEqual(loss_value, myconfig.TRIPLET_ALPHA)

    def test_get_triplet_loss2(self):
        anchor = torch.tensor([0.6, 0.8])
        pos = torch.tensor([0.6, 0.8])
        neg = torch.tensor([-0.8, 0.6])
        loss = neural_net.get_triplet_loss(anchor, pos, neg)
        loss_value = loss.data.numpy().item()
        self.assertAlmostEqual(loss_value, 0)

    def test_get_triplet_loss2(self):
        anchor = torch.tensor([0.6, 0.8])
        pos = torch.tensor([-0.8, 0.6])
        neg = torch.tensor([0.6, 0.8])
        loss = neural_net.get_triplet_loss(anchor, pos, neg)
        loss_value = loss.data.numpy().item()
        self.assertAlmostEqual(loss_value, 1 + myconfig.TRIPLET_ALPHA)

    def test_get_triplet_loss_from_batch_output1(self):
        batch_output = torch.tensor([[0.6, 0.8], [-0.8, 0.6], [0.6, 0.8]])
        loss = neural_net.get_triplet_loss_from_batch_output(
            batch_output, batch_size=1)
        loss_value = loss.data.numpy().item()
        self.assertAlmostEqual(loss_value, 1 + myconfig.TRIPLET_ALPHA)

    def test_get_triplet_loss_from_batch_output1(self):
        batch_output = torch.tensor(
            [[0.6, 0.8], [-0.8, 0.6], [0.6, 0.8],
             [0.6, 0.8], [-0.8, 0.6], [0.6, 0.8]])
        loss = neural_net.get_triplet_loss_from_batch_output(
            batch_output, batch_size=2)
        loss_value = loss.data.numpy().item()
        self.assertAlmostEqual(loss_value, 1 + myconfig.TRIPLET_ALPHA)

    def test_train_network(self):
        losses = neural_net.train_network(num_steps=2)
        self.assertEqual(len(losses), 2)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.encoder = neural_net.SpeakerEncoder()

    def test_run_inference(self):
        features = feature_extraction.extract_features(os.path.join(
            myconfig.TEST_DATA_DIR, "61/70968/61-70968-0000.flac"))
        embedding = evaluation.run_inference(features, self.encoder)
        self.assertEqual(embedding.shape, (myconfig.LSTM_HIDDEN_SIZE,))


if __name__ == "__main__":
    unittest.main()
