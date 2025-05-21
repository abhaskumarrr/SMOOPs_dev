import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_channels=16, lstm_hidden=64, lstm_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden, output_size)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.cnn(x).squeeze(-1)  # (batch, cnn_channels)
        x = x.unsqueeze(1)  # (batch, 1, cnn_channels)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out 