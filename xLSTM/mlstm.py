import torch
import torch.nn as nn
import torch.nn.functional as F

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstms = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for _ in range(num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        self.exp_input_gates = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_layers)])
        self.exp_forget_gates = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_layers)])
        self.output_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        self.reset_parameters()

    def reset_parameters(self):
        for lstm in self.lstms:
            nn.init.xavier_uniform_(lstm.weight_ih)
            nn.init.xavier_uniform_(lstm.weight_hh)
            nn.init.zeros_(lstm.bias_ih) 
            nn.init.zeros_(lstm.bias_hh)
        
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias) 
        nn.init.zeros_(self.W_v.bias)
        
        for gate in self.exp_input_gates + self.exp_forget_gates + self.output_gates:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, input_seq, hidden_state=None):
        batch_size = input_seq.size(0)
        seq_length = input_seq.size(1)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        output_seq = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            queries = self.W_q(x)
            keys = self.W_k(x)
            values = self.W_v(x)

            new_hidden_state = []
            for i, (lstm, dropout, i_gate, f_gate, o_gate) in enumerate(zip(self.lstms, self.dropout_layers, self.exp_input_gates, self.exp_forget_gates, self.output_gates)):
                if hidden_state[i][0] is None:
                    h, C = lstm(x)
                else:
                    h, C = hidden_state[i]
                
                i = torch.exp(i_gate(x))
                f = torch.exp(f_gate(x))

                C_t = f * C + i * torch.matmul(values.unsqueeze(2), keys.unsqueeze(1)).squeeze(1)
                attn_output = torch.matmul(queries, C_t).squeeze(2)
                
                o = torch.sigmoid(o_gate(h))
                h = o * attn_output
                new_hidden_state.append((h, C_t))
                
                if i < self.num_layers - 1:
                    x = dropout(h)
                else:
                    x = h
            hidden_state = new_hidden_state
            output_seq.append(x)

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq, hidden_state

    def init_hidden(self, batch_size):
        hidden_state = []
        for lstm in self.lstms:
            h = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=lstm.weight_ih.device)
            hidden_state.append((h, C))
        return hidden_state