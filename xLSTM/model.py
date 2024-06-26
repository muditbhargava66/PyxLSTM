"""
xLSTM: Extended Long Short-Term Memory Model

This module implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM model combines sLSTM and mLSTM blocks in a residual architecture
to achieve state-of-the-art performance on various language modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
from .block import xLSTMBlock

class xLSTM(nn.Module):
    """
    xLSTM model implementation.

    This model uses a combination of sLSTM and mLSTM blocks in a residual architecture.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_size (int): Size of the token embeddings.
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.0.
        lstm_type (str, optional): Type of LSTM to use ('slstm' or 'mlstm'). Default: 'slstm'.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lstm_type = lstm_type

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.ModuleList([
            xLSTMBlock(embedding_size, hidden_size, num_layers, dropout, lstm_type)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the xLSTM model.

        Args:
            input_seq (Tensor): Input sequence of token indices.
            hidden_states (list of tuples, optional): Initial hidden states for each block. Default: None.

        Returns:
            tuple: Output logits and final hidden states.
        """
        embedded_seq = self.embedding(input_seq)
        
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks
        
        output_seq = embedded_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])
        
        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states