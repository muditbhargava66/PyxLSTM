import torch
from torch.utils.data import Dataset

class LanguageModelingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        with open(data_path, "r", encoding="utf-8") as file:
            text = file.read()
        tokens = self.tokenizer.encode(text)
        return tokens

    def __len__(self):
        return len(self.data) - self.max_length

    def __getitem__(self, idx):
        input_ids = self.data[idx : idx + self.max_length]
        target_ids = self.data[idx + 1 : idx + self.max_length + 1]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        return input_ids, target_ids