# Lung and Colon Cancer Image Analysis

This project implements a comprehensive deep learning approach for lung and colon cancer classification using microscopy images. The project includes training a custom three-layer CNN from scratch and fine-tuning pre-trained ResNet and EfficientNet models using transfer learning, followed by performance comparison and analysis.

## Dataset Structure

```
dataset/
├── colon_image_sets/
│   ├── colon_aca/    # Colon adenocarcinoma images
│   └── colon_n/      # Normal colon tissue images
└── lung_image_sets/
    ├── lung_aca/     # Lung adenocarcinoma images
    ├── lung_n/       # Normal lung tissue images
    └── lung_scc/     # Lung squamous cell carcinoma images
```

## Project Structure

```
.
├── dataset/          # Contains all image datasets
├── src/
│   ├── analysis/
│   │   ├── plots/    # Generated visualization outputs
│   │   ├── eda_colon.py  # Colon cancer image analysis
│   │   └── eda_lungs.py  # Lung cancer image analysis
│   └── models/
│       ├── cnn/                  # Baseline CNN model code and checkpoints
│       ├── transfer_learning/
│       │   ├── EfficientNet/     # EfficientNet transfer learning code and checkpoints
│       │   └── ResNet/           # ResNet transfer learning code and checkpoints
│       ├── dataset.py            # Dataset and DataLoader implementation
│       ├── cnn_model.py          # Custom CNN architecture
│       ├── train.py              # Training and evaluation script for CNN
│       └── compare_models.py     # Script to compare all models
└── requirements.txt  # Python dependencies
```

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Trained Models

This project trains and compares the following models:

- **Custom 3-layer CNN** (from scratch, see `cnn/`)
- **EfficientNet (transfer learning)** (see `transfer_learning/EfficientNet/`)
- **ResNet (transfer learning)** (see `transfer_learning/ResNet/`)

Each model is trained separately for both lung and colon cancer datasets.

## Model Training Scripts

### Baseline CNN

To train the baseline CNN:

```bash
cd src/models/cnn
python train.py
```

### EfficientNet Transfer Learning

To train EfficientNet:

```bash
cd src/models/transfer_learning/EfficientNet
python train_efficientnet.py
```

### ResNet Transfer Learning

To train ResNet:

```bash
cd src/models/transfer_learning/ResNet
python train_resnet.py
```

Each script will:

- Train on both lung and colon datasets
- Save best model checkpoints
- Save training metrics as JSON
- Generate training/validation metric plots

## Model Comparison

After training, you can compare all models using:

```bash
cd src/models
python compare_models.py
```

This will generate comparison plots for both lung and colon cancer classification in `compare_plots/`:

- `lung_model_comparison.png`
- `colon_model_comparison.png`

Each plot shows all three models (CNN, EfficientNet, ResNet) compared on:

- Validation Accuracy
- Validation Sensitivity
- Validation Specificity
- Validation Loss

**Example:**

![Lung Model Comparison](src/models/compare_plots/lung_model_comparison.png)
![Colon Model Comparison](src/models/compare_plots/colon_model_comparison.png)

## Output Files

Each model's checkpoints and metrics are saved in their respective `checkpoints/` directories:

- `cnn/checkpoints/`
  - `lung_model.pth`, `lung_metrics.json`, plots
  - `colon_model.pth`, `colon_metrics.json`, plots
- `transfer_learning/EfficientNet/checkpoints/`
  - `lung_efficientnet.pth`, `lung_efficientnet_metrics.json`, plots
  - `colon_efficientnet.pth`, `colon_efficientnet_metrics.json`, plots
- `transfer_learning/ResNet/checkpoints/`
  - `lung_resnet.pth`, `lung_resnet_metrics.json`, plots
  - `colon_resnet.pth`, `colon_resnet_metrics.json`, plots

## Model Performance Metrics

The models are evaluated using:

- **Accuracy:** Overall classification accuracy
- **Sensitivity:** True positive rate (cancer detection rate)
- **Specificity:** True negative rate (normal tissue detection rate)
