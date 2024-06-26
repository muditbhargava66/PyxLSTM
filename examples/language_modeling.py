import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xLSTM import xLSTM
import time
import math

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Define hyperparameters
vocab_size = 1000
embedding_size = 128
hidden_size = 256
num_layers = 1
num_blocks = 2
batch_size = 64
seq_length = 20
num_epochs = 5
learning_rate = 0.0001
clip_value = 1.0

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_length, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks).to(device)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Embedding]:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = DummyDataset(vocab_size, seq_length, 1000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_seq = batch[:, :-1].to(device)
        target_seq = batch[:, 1:].to(device)
        
        if check_nan(input_seq, "input_seq"):
            break
        
        output, _ = model(input_seq)
        
        if check_nan(output, "model output"):
            break
        
        output = output.contiguous().view(-1, vocab_size)
        target_seq = target_seq.contiguous().view(-1)
        
        loss = criterion(output, target_seq)
        
        if check_nan(loss, "loss"):
            break
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_nan(param.grad, f"gradient of {name}"):
                    break
        
        optimizer.step()
        
        # Check parameters after update
        for name, param in model.named_parameters():
            if check_nan(param, f"parameter {name} after update"):
                break
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    if math.isnan(total_loss):
        print("NaN detected in total_loss. Stopping training.")
        break
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

end_time = time.time()
print(f"Training completed! Total time: {end_time - start_time:.2f} seconds")