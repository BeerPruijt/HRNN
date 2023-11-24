import torch


train_sequences = create_inout_sequences(train_data, sequence_length)
test_sequences = create_inout_sequences(test_data, sequence_length)

# Convert to PyTorch tensors
train_sequences = [torch.tensor(s, dtype=torch.float32) for s in train_sequences]
test_sequences = [torch.tensor(s, dtype=torch.float32) for s in test_sequences]

import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Create the GRU model
model = GRUModel(input_size=1, hidden_size=gru_params['hidden_size'], num_layers=gru_params['num_layers'], dropout=gru_params['dropout'])
