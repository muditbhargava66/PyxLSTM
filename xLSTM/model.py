import torch
import torch.nn as nn
from .block import xLSTMBlock

class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.ModuleList([xLSTMBlock(embedding_size if i == 0 else hidden_size,
                                                hidden_size, num_layers, dropout, bidirectional, lstm_type)
                                     for i in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_seq, hidden_states=None):
        embedded_seq = self.embedding(input_seq)

        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = embedded_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state = block(output_seq, hidden_states[i])
            if self.lstm_type == "slstm":
                hidden_states[i] = [[hidden_state[j][0], hidden_state[j][1]] for j in range(len(hidden_state))]
            else:
                hidden_states[i] = hidden_state

        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states