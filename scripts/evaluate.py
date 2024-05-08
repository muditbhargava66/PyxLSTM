import argparse
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from xLSTM import xLSTM
from data import Dataset, Tokenizer
from utils import set_seed, load_checkpoint

def evaluate(model, dataloader, criterion, device):
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(args.vocab_file)
    test_dataset = Dataset(args.test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = xLSTM(len(tokenizer), args.embedding_size, args.hidden_size,
                  args.num_layers, args.num_blocks, args.dropout, args.bidirectional, args.lstm_type)
    model.to(device)

    load_checkpoint(model, None, args.checkpoint_path)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    test_loss = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {math.exp(test_loss):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--lstm_type", type=str, default="slstm", help="LSTM type (slstm or mlstm)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)