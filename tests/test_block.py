import unittest
import torch
from xLSTM.block import xLSTMBlock

class TestXLSTMBlock(unittest.TestCase):
    def setUp(self):
        self.input_size = 64
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.1
        self.batch_size = 4
        self.seq_length = 6

    def test_forward_pass_slstm(self):
        xlstm_block = xLSTMBlock(self.input_size, self.hidden_size, self.num_layers, self.dropout, lstm_type="slstm")
        input_seq = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output_seq, hidden_state = xlstm_block(input_seq)

        self.assertEqual(output_seq.shape, (self.batch_size, self.seq_length, self.input_size))
        self.assertEqual(len(hidden_state), self.num_layers)
        self.assertEqual(hidden_state[0][0].shape, (self.batch_size, self.hidden_size))
        self.assertEqual(hidden_state[0][1].shape, (self.batch_size, self.hidden_size))

    def test_forward_pass_mlstm(self):
        xlstm_block = xLSTMBlock(self.input_size, self.hidden_size, self.num_layers, self.dropout, lstm_type="mlstm")
        input_seq = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output_seq, hidden_state = xlstm_block(input_seq)

        self.assertEqual(output_seq.shape, (self.batch_size, self.seq_length, self.input_size))
        self.assertEqual(len(hidden_state), self.num_layers)
        self.assertEqual(hidden_state[0][0].shape, (self.batch_size, self.hidden_size))
        self.assertEqual(hidden_state[0][1].shape, (self.batch_size, self.hidden_size, self.hidden_size))

    def test_backward_pass(self):
        xlstm_block = xLSTMBlock(self.input_size, self.hidden_size, self.num_layers, self.dropout)
        input_seq = torch.randn(self.batch_size, self.seq_length, self.input_size, requires_grad=True)
        output_seq, _ = xlstm_block(input_seq)

        target = torch.randn(self.batch_size, self.seq_length, self.input_size, requires_grad=True)
        loss = torch.nn.MSELoss()(output_seq, target)
        loss.backward()

        for param in xlstm_block.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()