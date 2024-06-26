import torch

from xLSTM import xLSTMBlock
from xLSTM import sLSTM
from xLSTM import mLSTM
from xLSTM import xLSTM

# Define the model hyperparameters
vocab_size = 10000
embedding_size = 256
hidden_size = 512
num_layers = 2
num_blocks = 3
dropout = 0.1
bidirectional = False
lstm_type = "slstm"  # or "mlstm"

# Create an instance of the xLSTM model
model = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
              dropout, bidirectional, lstm_type)

# Generate random input sequence
batch_size = 4
seq_length = 10
input_seq = torch.randint(0, vocab_size, (batch_size, seq_length))

# Forward pass
output_seq, hidden_states = model(input_seq)

# Print the shapes of the output and hidden states
print("Output sequence shape:", output_seq.shape)
print("Hidden states shapes:")
for i, hidden_state in enumerate(hidden_states):
    if lstm_type == "slstm":
        print(f"Block {i+1} - Hidden state shape: {hidden_state[0][0].shape}")
        print(f"Block {i+1} - Cell state shape: {hidden_state[0][1].shape}")
    else:
        print(f"Block {i+1} - Hidden state shape: {hidden_state[0].shape}")
        print(f"Block {i+1} - Cell state shape: {hidden_state[1].shape}")

# Test the sLSTM module
slstm = sLSTM(embedding_size, hidden_size, num_layers, dropout)
input_seq_slstm = torch.randn(batch_size, seq_length, embedding_size)
output_seq_slstm, hidden_state_slstm = slstm(input_seq_slstm)
print("\nsLSTM module:")
print("Output sequence shape:", output_seq_slstm.shape)
print("Hidden state shape:", hidden_state_slstm[0][0].shape)
print("Cell state shape:", hidden_state_slstm[0][1].shape)

# Test the mLSTM module
mlstm = mLSTM(embedding_size, hidden_size, num_layers, dropout)
input_seq_mlstm = torch.randn(batch_size, seq_length, embedding_size)
output_seq_mlstm, hidden_state_mlstm = mlstm(input_seq_mlstm)
print("\nmLSTM module:")
print("Output sequence shape:", output_seq_mlstm.shape)
print("Hidden state shape:", hidden_state_mlstm[0][0].shape)
print("Cell state shape:", hidden_state_mlstm[0][1].shape)