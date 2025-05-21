import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, log_dir='runs', patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')

    def train(self, num_epochs=100):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch[:, -1, 0].unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            val_loss = self.evaluate()
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        self.writer.close()

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, batch[:, -1, 0].unsqueeze(1))
                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        return val_loss 