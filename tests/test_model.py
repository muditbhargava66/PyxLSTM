"""
Unit tests for the xLSTM model.

This module contains unit tests for the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

These tests verify the forward and backward passes of the xLSTM model,
as well as the functionality of both sLSTM and mLSTM variants.

Author: Mudit Bhargava
Date: June 2024
"""

import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xLSTM.model import xLSTM

class TestXLSTMModel(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.vocab_size = 1000
        self.embedding_size = 64
        self.hidden_size = 64
        self.num_layers = 2
        self.num_blocks = 3
        self.batch_size = 4
        self.seq_length = 6

    def test_forward_pass(self):
        """Test the forward pass of the xLSTM model."""
        model = xLSTM(self.vocab_size, self.embedding_size, self.hidden_size,
                      self.num_layers, self.num_blocks)
        input_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output_seq, hidden_states = model(input_seq)

        self.assertEqual(output_seq.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.assertEqual(len(hidden_states), self.num_blocks)

    def test_backward_pass(self):
        """Test the backward pass of the xLSTM model."""
        model = xLSTM(self.vocab_size, self.embedding_size, self.hidden_size,
                      self.num_layers, self.num_blocks)
        input_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output_seq, _ = model(input_seq)
        loss = output_seq.sum()
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    def test_mlstm_model(self):
        """Test the xLSTM model with mLSTM blocks."""
        model = xLSTM(self.vocab_size, self.embedding_size, self.hidden_size,
                      self.num_layers, self.num_blocks, lstm_type="mlstm")
        input_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output_seq, hidden_states = model(input_seq)

        self.assertEqual(output_seq.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.assertEqual(len(hidden_states), self.num_blocks)
        self.assertEqual(hidden_states[0][0][0].shape, (self.batch_size, self.hidden_size))
        self.assertEqual(hidden_states[0][0][1].shape, (self.batch_size, self.hidden_size, self.hidden_size))

if __name__ == '__main__':
    unittest.main()