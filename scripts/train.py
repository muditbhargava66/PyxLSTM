import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from xLSTM import xLSTM
from data import Dataset, Tokenizer
from utils import set_seed, save_checkpoint, load_checkpoint, log_metrics

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(args.vocab_file)
    train_dataset = Dataset(args.train_data, tokenizer)
    valid_dataset = Dataset(args.valid_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = xLSTM(len(tokenizer), args.embedding_size, args.hidden_size,
                  args.num_layers, args.num_blocks, args.dropout, args.bidirectional, args.lstm_type)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_valid_loss = float("inf")
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        valid_loss = validate(model, valid_dataloader, criterion, device)
        log_metrics(epoch, train_loss, valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--lstm_type", type=str, default="slstm", help="LSTM type (slstm or mlstm)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt", help="Path to save checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)