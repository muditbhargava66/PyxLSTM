import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    """
    A single LSTM block in the xLSTM model.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional, lstm_type):
        """
        Initialize the LSTM block.

        Args:
            input_dim (int): The input dimension of the LSTM block.
            hidden_dim (int): The hidden dimension of the LSTM block.
            num_layers (int): The number of LSTM layers.
            dropout (float): The dropout rate.
            bidirectional (bool): Whether the LSTM is bidirectional.
            lstm_type (str): The type of LSTM (e.g., "slstm").
        """
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.lstm_type = lstm_type

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass through the LSTM block.

        Args:
            input_seq (torch.Tensor): The input sequence.
            hidden_state (torch.Tensor, optional): The initial hidden state.

        Returns:
            output_seq (torch.Tensor): The output sequence.
            hidden_state (torch.Tensor): The final hidden state.
        """
        output_seq, hidden_state = self.lstm(input_seq, hidden_state)
        return output_seq, hidden_state

class xLSTM(nn.Module):
    """
    The xLSTM model.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_blocks,
                 dropout=0.0, bidirectional=False, lstm_type="slstm"):
        """
        Initialize the xLSTM model.

        Args:
            vocab_size (int): The vocabulary size.
            embedding_dim (int): The embedding dimension.
            hidden_dim (int): The hidden dimension.
            num_layers (int): The number of LSTM layers.
            num_blocks (int): The number of LSTM blocks.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to False.
            lstm_type (str, optional): The type of LSTM. Defaults to "slstm".
        """
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_blocks = nn.ModuleList([LSTMBlock(embedding_dim if i == 0 else hidden_dim,
                                                    hidden_dim, num_layers, dropout, bidirectional, lstm_type)
                                          for i in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass through the xLSTM model.

        Args:
            input_seq (torch.Tensor): The input sequence.
            hidden_states (list of torch.Tensor, optional): The initial hidden states.

        Returns:
            output_seq (torch.Tensor): The output sequence.
            hidden_states (list of torch.Tensor): The final hidden states.
        """
        embedded_seq = self.embedding_layer(input_seq)

        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = embedded_seq
        for i, block in enumerate(self.lstm_blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])

        output_seq = self.output_layer(output_seq[:, -1, :])  # Take the last hidden state
        return output_seq, hidden_states