from collections import defaultdict

class Tokenizer:
    def __init__(self, vocab_file, special_tokens=None):
        self.vocab_file = vocab_file
        self.special_tokens = special_tokens or []
        self.word2idx = {}
        self.idx2word = {}
        self._build_vocab()

    def _build_vocab(self):
        with open(self.vocab_file, "r", encoding="utf-8") as file:
            vocab = [line.strip() for line in file]

        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.idx2word = defaultdict(lambda: "<unk>")

        for token in self.special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token

        for word in vocab:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word

        self.unk_token = "<unk>"
        self.unk_token_id = self.word2idx[self.unk_token]
        self.pad_token = "<pad>"
        self.pad_token_id = self.word2idx[self.pad_token]

    def encode(self, text):
        tokens = text.split()
        ids = [self.word2idx[token] for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.idx2word[idx] for idx in ids]
        text = " ".join(tokens)
        return text

    def __len__(self):
        return len(self.word2idx)