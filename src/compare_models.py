import json
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to metrics files
cnn_metrics_dir = Path('models/cnn/checkpoints')
effnet_metrics_dir = Path('models/transfer_learning/EfficientNet/checkpoints')
resnet_metrics_dir = Path('models/transfer_learning/ResNet/checkpoints')

models = {
    'CNN': cnn_metrics_dir,
    'EfficientNet': effnet_metrics_dir,
    'ResNet': resnet_metrics_dir
}

cancer_types = ['lung', 'colon']
metrics = ['val_accuracy', 'val_sensitivity', 'val_specificity', 'val_loss']
metric_titles = {
    'val_accuracy': 'Validation Accuracy',
    'val_sensitivity': 'Validation Sensitivity',
    'val_specificity': 'Validation Specificity',
    'val_loss': 'Validation Loss'
}

colors = {
    'CNN': 'b',
    'EfficientNet': 'g',
    'ResNet': 'r'
}

for cancer_type in cancer_types:
    # Load metrics for each model
    model_histories = {}
    for model_name, metrics_dir in models.items():
        if model_name == 'CNN':
            metrics_path = metrics_dir / f'{cancer_type}_metrics.json'
        elif model_name == 'EfficientNet':
            metrics_path = metrics_dir / f'{cancer_type}_efficientnet_metrics.json'
        elif model_name == 'ResNet':
            metrics_path = metrics_dir / f'{cancer_type}_resnet_metrics.json'
        else:
            continue
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} not found. Skipping {model_name} for {cancer_type}.")
            continue
        with open(metrics_path, 'r') as f:
            model_histories[model_name] = json.load(f)
    # Plot comparison for each metric
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for model_name, history in model_histories.items():
            y = history[metric]
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, label=model_name, color=colors[model_name])
        ax.set_title(f'{metric_titles[metric]} ({cancer_type.capitalize()})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_titles[metric])
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    out_dir = Path('compare_plots')
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f'{cancer_type}_model_comparison.png')
    plt.close()
    print(f'Saved comparison plot for {cancer_type} to {out_dir / f"{cancer_type}_model_comparison.png"}') 