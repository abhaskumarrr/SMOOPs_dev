import torch
import torch.nn as nn
import torch.optim as optim
from ml.backend.src.data.loader import get_dataloader
from ml.backend.src.models.cnn_lstm import CNNLSTMModel
from ml.backend.src.training.trainer import Trainer
import os

# Config
csv_file = 'sample_data/BTCUSD_15m.csv'
schema_file = 'ml/backend/src/data/schema.yaml'
batch_size = 64
seq_len = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare DataLoader
train_loader = get_dataloader(csv_file, schema_file, batch_size=batch_size, shuffle=True, seq_len=seq_len)
val_loader = get_dataloader(csv_file, schema_file, batch_size=batch_size, shuffle=False, seq_len=seq_len)

# Model params (input_size = number of features excluding timestamp)
sample = next(iter(train_loader))
print('Sample batch shape:', sample.shape)  # (batch, seq_len, input_size)
input_size = sample.shape[2]
model = CNNLSTMModel(input_size=input_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Trainer
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, log_dir='runs/cnnlstm')
trainer.train(num_epochs=20)

# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/cnnlstm_trained.pt')
print('Training complete. Model saved to models/cnnlstm_trained.pt') 