import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    """
    A stacked LSTM model.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        """
        Initialize the stacked LSTM model.

        Args:
            input_dim (int): The input dimension of the LSTM model.
            hidden_dim (int): The hidden dimension of the LSTM model.
            num_layers (int): The number of LSTM layers.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super(StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Initialize the LSTM cells for each layer
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_dim, hidden_dim) for _ in range(num_layers)])

        # Initialize the dropout layers for each layer (except the last one)
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass through the stacked LSTM model.

        Args:
            input_seq (torch.Tensor): The input sequence.
            hidden_state (list of torch.Tensor, optional): The initial hidden state.

        Returns:
            output_seq (torch.Tensor): The output sequence.
            hidden_state (list of torch.Tensor): The final hidden state.
        """
        batch_size = input_seq.size(0)
        seq_length = input_seq.size(1)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        output_seq = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for i, (lstm, dropout) in enumerate(zip(self.lstm_cells, self.dropout_layers)):
                h, c = lstm(x, hidden_state[i])
                hidden_state[i] = (h, c)
                if i < self.num_layers - 1:
                    x = dropout(h)
                else:
                    x = h
            output_seq.append(x)

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq, hidden_state

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state.

        Args:
            batch_size (int): The batch size.

        Returns:
            hidden_state (list of torch.Tensor): The initial hidden state.
        """
        hidden_state = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, device=self.lstm_cells[0].weight_ih.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=self.lstm_cells[0].weight_ih.device)
            hidden_state.append((h, c))
        return hidden_state