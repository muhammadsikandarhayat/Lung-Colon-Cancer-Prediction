# Lung and Colon Cancer Image Analysis

This project provides exploratory data analysis (EDA) tools for analyzing lung and colon cancer microscopy images. The dataset includes images of different types of lung and colon tissues, both normal and cancerous.

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
│   └── analysis/
│       ├── eda_colon.py  # Colon cancer image analysis
│       └── eda_lungs.py  # Lung cancer image analysis
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
   pip install numpy>=1.24.0 opencv-python>=4.8.0 matplotlib>=3.7.0 pandas>=2.0.0
   ```

## Analysis Scripts

### Colon Cancer Analysis (`eda_colon.py`)

This script analyzes colon cancer images and generates:

- Class distribution statistics
- RGB channel intensity distributions
- Basic image statistics (mean, std, min, max values)
- Image dimension analysis

To run:

```bash
python src/analysis/eda_colon.py
```

### Lung Cancer Analysis (`eda_lungs.py`)

This script provides analysis for lung cancer images including:

- Distribution analysis across three classes (ACA, Normal, SCC)
- RGB channel intensity distributions
- Sample image visualization
- Basic image statistics
- Image dimension analysis

To run:

```bash
python src/analysis/eda_lungs.py
```

## Output

Both scripts generate:

1. Statistical summaries printed to console
2. Visualization plots saved as PNG files:
   - Class distribution plots
   - RGB intensity distribution plots
   - Sample images (lung dataset only)

## Generated Visualizations

### Colon Cancer Dataset

- `colon_class_distribution.png`: Bar plot showing the distribution of images across classes
- `colon_intensity_distribution.png`: RGB channel intensity distributions for each class

### Lung Cancer Dataset

- `lung_class_distribution.png`: Bar plot showing the distribution of images across three classes
- `lung_intensity_distribution.png`: RGB channel intensity distributions for each class
- `lung_sample_images.png`: Sample images from each class

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- OpenCV >= 4.8.0
- Matplotlib >= 3.7.0
- Pandas >= 2.0.0
