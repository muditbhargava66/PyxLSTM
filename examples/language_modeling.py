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
    train(model, train_dataset, valid_dataset, optimizer, criterion, config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    main(args)