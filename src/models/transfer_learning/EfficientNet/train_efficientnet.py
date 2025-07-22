import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from dataset import CancerDataset, get_data_transforms

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model_state):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_state)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_state)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_state):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss
        self.best_model_state = model_state.copy()

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def plot_metrics(metrics_history, save_path):
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    ax1.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax3.plot(epochs, metrics_history['val_sensitivity'], 'g-', label='Validation Sensitivity')
    ax3.set_title('Validation Sensitivity')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sensitivity')
    ax3.legend()
    ax4.plot(epochs, metrics_history['val_specificity'], 'y-', label='Validation Specificity')
    ax4.set_title('Validation Specificity')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Specificity')
    ax4.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics_history, save_path):
    with open(save_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=7):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_sensitivity': [],
        'val_specificity': []
    }
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        metrics_history['train_loss'].append(epoch_loss)
        model.eval()
        val_predictions = []
        val_targets = []
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                _, preds = torch.max(outputs, 1)
                val_predictions.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        val_epoch_loss = val_running_loss / len(val_loader)
        metrics_history['val_loss'].append(val_epoch_loss)
        val_metrics = calculate_metrics(val_targets, val_predictions)
        metrics_history['val_accuracy'].append(val_metrics['accuracy'])
        metrics_history['val_sensitivity'].append(val_metrics['sensitivity'])
        metrics_history['val_specificity'].append(val_metrics['specificity'])
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_epoch_loss:.4f}')
        print(f'Validation Metrics:')
        print(f'  Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'  Sensitivity: {val_metrics["sensitivity"]:.4f}')
        print(f'  Specificity: {val_metrics["specificity"]:.4f}')
        print('-' * 60)
        early_stopping(val_epoch_loss, model.state_dict())
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    return early_stopping.best_model_state, metrics_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    data_dir = Path('../../../../dataset')
    cancer_types = ['lung', 'colon']
    batch_size = 32
    learning_rate = 0.0005
    num_epochs = 30
    patience = 7
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    for cancer_type in cancer_types:
        print(f'\nTraining EfficientNet for {cancer_type} cancer classification')
        print('=' * 60)
        data_transforms = get_data_transforms()
        dataset = CancerDataset(
            data_dir=data_dir,
            cancer_type=cancer_type,
            transform=data_transforms['train']
        )
        print(f"\nDataset size: {len(dataset)} images")
        if len(dataset) == 0:
            raise ValueError(f"No images found for {cancer_type} cancer dataset!")
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        print(f"Training set size: {train_size}")
        print(f"Validation set size: {val_size}")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_dataset.dataset.transform = data_transforms['val']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        # Load EfficientNet
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_model_state, metrics_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            patience=patience
        )
        model_path = save_dir / f'{cancer_type}_efficientnet.pth'
        torch.save(best_model_state, model_path)
        print(f'\nBest model saved to {model_path}')
        metrics_path = save_dir / f'{cancer_type}_efficientnet_metrics.json'
        save_metrics(metrics_history, metrics_path)
        print(f'Metrics saved to {metrics_path}')
        plot_path = plots_dir / f'{cancer_type}_efficientnet_training_metrics.png'
        plot_metrics(metrics_history, plot_path)
        print(f'Training plots saved to {plot_path}')

if __name__ == '__main__':
    main() 