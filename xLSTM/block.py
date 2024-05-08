import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBlock(nn.Module):
    """
    A single LSTM block in the xLSTM model.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0, bidirectional=False, lstm_type="slstm"):
        """
        Initialize the LSTM block.

        Args:
            input_dim (int): The input dimension of the LSTM block.
            hidden_dim (int): The hidden dimension of the LSTM block.
            num_layers (int): The number of LSTM layers.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to False.
            lstm_type (str, optional): The type of LSTM (e.g., "slstm" or "mlstm"). Defaults to "slstm".
        """
        super(LSTMBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        # Initialize the LSTM layer based on the LSTM type
        if lstm_type == "slstm":
            from .slstm import sLSTM
            self.lstm_layer = sLSTM(input_dim, hidden_dim, num_layers, dropout)
        elif lstm_type == "mlstm":
            from .mlstm import mLSTM
            self.lstm_layer = mLSTM(input_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")

        # Initialize the linear layer to transform the LSTM output
        self.linear_layer = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)

        # Initialize the layer normalization layer
        self.normalization_layer = nn.LayerNorm(hidden_dim)

        # Initialize the dropout layer
        self.dropout_layer = nn.Dropout(dropout)

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
        # Pass the input sequence through the LSTM layer
        lstm_output, hidden_state = self.lstm_layer(input_seq, hidden_state)

        # If the LSTM is bidirectional, concatenate the forward and backward outputs
        if self.bidirectional:
            lstm_output = torch.cat((lstm_output[:, :, :self.hidden_dim], lstm_output[:, :, self.hidden_dim:]), dim=-1)

        # Transform the LSTM output using the linear layer
        output = self.linear_layer(lstm_output)

        # Add the input sequence to the output and apply layer normalization
        output = self.normalization_layer(output + input_seq)

        # Apply dropout to the output
        output = self.dropout_layer(output)

        return output, hidden_state