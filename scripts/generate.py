import argparse
import torch
from xLSTM import xLSTM
from data import Tokenizer
from utils import set_seed, load_checkpoint

def generate(model, tokenizer, prompt, max_length, device):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = []
    hidden_states = None
    with torch.no_grad():
        for _ in range(max_length):
            outputs, hidden_states = model(input_ids, hidden_states)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_ids.append(next_token_id)
            input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

    generated_text = tokenizer.decode(generated_ids)
    return generated_text

def main(args):
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(args.vocab_file)

    model = xLSTM(len(tokenizer), args.embedding_size, args.hidden_size,
                  args.num_layers, args.num_blocks, args.dropout, args.bidirectional, args.lstm_type)
    model.to(device)

    load_checkpoint(model, None, args.checkpoint_path)

    generated_text = generate(model, tokenizer, args.prompt, args.max_length, device)
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--lstm_type", type=str, default="slstm", help="LSTM type (slstm or mlstm)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)