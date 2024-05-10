import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from xLSTM.model import xLSTM
from xLSTM.data import LanguageModelingDataset, Tokenizer
from xLSTM.utils import load_config, set_seed, get_device

def main(args):
    """
    Main function to train the xLSTM model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    # Load the configuration file
    config = load_config(args.config)
    print(f"Loaded configuration file: {args.config}")

    # Set the random seed for reproducibility
    set_seed(config.seed)
    print(f"Set random seed: {config.seed}")

    # Get the device (GPU or CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Initialize the tokenizer
    tokenizer = Tokenizer(config.vocab_file)
    print(f"Loaded tokenizer from: {config.vocab_file}")

    # Load the training and validation datasets
    train_dataset = LanguageModelingDataset(config.train_data, tokenizer, config.max_length)
    print(f"Loaded training dataset from: {config.train_data}")
    valid_dataset = LanguageModelingDataset(config.valid_data, tokenizer, config.max_length)
    print(f"Loaded validation dataset from: {config.valid_data}")

    # Initialize the xLSTM model
    model = xLSTM(len(tokenizer), config.embedding_size, config.hidden_size,
                  config.num_layers, config.num_blocks, config.dropout,
                  config.bidirectional, config.lstm_type)
    model.to(device)
    print(f"Initialized xLSTM model with {config.num_layers} layers and {config.num_blocks} blocks")

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print(f"Initialized optimizer with learning rate: {config.learning_rate}")

    # Train the model
    def train(model, train_dataset, valid_dataset, optimizer, criterion, config, device):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, len(tokenizer)), targets.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs, _ = model(inputs)
                loss = criterion(outputs.view(-1, len(tokenizer)), targets.view(-1))
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    main(args)