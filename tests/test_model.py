import unittest
import torch
from xLSTM.model import xLSTM

class TestXLSTMModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_size = 64
        self.hidden_size = 64
        self.num_layers = 2
        self.num_blocks = 3
        self.dropout = 0.1
        self.batch_size = 4
        self.seq_length = 6

    def test_forward_pass(self):
        xlstm_model = xLSTM(self.vocab_size, self.embedding_size, self.hidden_size,
                            self.num_layers, self.num_blocks, self.dropout)
        input_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output_seq, hidden_states = xlstm_model(input_seq)

        self.assertEqual(output_seq.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.assertEqual(len(hidden_states), self.num_blocks)

    def test_backward_pass(self):
        xlstm_model = xLSTM(self.vocab_size, self.embedding_size, self.hidden_size,
                            self.num_layers, self.num_blocks, self.dropout)
        input_seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output_seq, _ = xlstm_model(input_seq)

        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss = torch.nn.CrossEntropyLoss()(output_seq.view(-1, self.vocab_size), target.view(-1))
        loss.backward()

        for param in xlstm_model.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()