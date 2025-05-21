import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        out = self.transformer(x)
        out = self.fc(out[-1])
        return out 