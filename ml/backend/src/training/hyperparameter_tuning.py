import optuna
import torch
from torch.utils.data import DataLoader
from .trainer import Trainer

def objective(trial, model_class, dataset, val_dataset, device):
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    model = model_class(input_size=dataset[0].shape[-1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True))
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs=30)
    val_loss = trainer.evaluate()
    return val_loss

def run_optuna(model_class, dataset, val_dataset, device, n_trials=20):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_class, dataset, val_dataset, device), n_trials=n_trials)
    print('Best trial:', study.best_trial.params)
    return study.best_trial 