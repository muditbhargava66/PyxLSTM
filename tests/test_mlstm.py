"""
Unit tests for the mLSTM (Matrix LSTM) module.

This module contains unit tests for the mLSTM as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

These tests verify the forward and backward passes of the mLSTM,
ensuring correct shapes and gradient flow.

Author: Mudit Bhargava
Date: June 2024
"""

import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xLSTM.mlstm import mLSTM

class TestMLSTM(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters and initialize mLSTM."""
        self.input_size = 64
        self.hidden_size = 64
        self.num_layers = 2
        self.batch_size = 4
        self.seq_length = 6
        self.mlstm = mLSTM(self.input_size, self.hidden_size, self.num_layers)

    def test_forward_pass(self):
        """Test the forward pass of the mLSTM."""
        input_seq = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output_seq, hidden_state = self.mlstm(input_seq)

        self.assertEqual(output_seq.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.assertEqual(len(hidden_state), self.num_layers)
        self.assertEqual(hidden_state[0][0].shape, (self.batch_size, self.hidden_size))
        self.assertEqual(hidden_state[0][1].shape, (self.batch_size, self.hidden_size, self.hidden_size))

    def test_backward_pass(self):
        """Test the backward pass of the mLSTM."""
        input_seq = torch.randn(self.batch_size, self.seq_length, self.input_size, requires_grad=True)
        output_seq, _ = self.mlstm(input_seq)
        loss = output_seq.sum()
        loss.backward()

        self.assertIsNotNone(input_seq.grad)
        for param in self.mlstm.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()