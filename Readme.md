# Lung and Colon Cancer Image Analysis

This project implements a comprehensive deep learning approach for lung and colon cancer classification using microscopy images. The project includes training a custom three-layer CNN from scratch and fine-tuning pre-trained ResNet50 and EfficientNet models, followed by performance comparison and analysis.

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
│       ├── plots/    # Generated visualization outputs
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
   pip install -r requirements.txt
   ```

## Analysis Scripts

### Colon Cancer Analysis (`eda_colon.py`)

This script analyzes colon cancer images and generates:

- Class distribution statistics
- RGB and HSV color space analysis
- Image quality metrics:
  - Blur detection
  - Contrast measurement
  - Brightness analysis
- Texture analysis using GLCM features:
  - Contrast
  - Dissimilarity
  - Homogeneity
  - Energy
  - Correlation
- Basic image statistics (mean, std, min, max values)
- Image dimension and aspect ratio analysis
- File size distribution

To run:

```bash
python src/analysis/eda_colon.py
```

### Lung Cancer Analysis (`eda_lungs.py`)

This script provides analysis for lung cancer images including:

- Distribution analysis across three classes (ACA, Normal, SCC)
- RGB and HSV color space analysis
- Image quality metrics
- Texture feature analysis
- Sample image visualization
- Basic image statistics
- Image dimension and aspect ratio analysis
- File size distribution

To run:

```bash
python src/analysis/eda_lungs.py
```

## Output

Both scripts generate:

1. Statistical summaries printed to console:
   - Class distribution
   - Image dimensions
   - Aspect ratio statistics
   - File size statistics
2. Visualization plots saved in `src/analysis/plots/`:
   - Class distribution plots
   - RGB intensity distribution plots
   - Quality metrics distributions
   - Texture feature distributions

## Generated Visualizations

### Colon Cancer Dataset

All plots are saved in `src/analysis/plots/`:

- `colon_class_distribution.png`: Bar plot showing the distribution of images across classes
- `colon_intensity_distribution.png`: RGB channel intensity distributions for each class
- `colon_quality_metrics.png`: Distribution of blur, contrast, and brightness metrics
- `colon_texture_features.png`: Distribution of GLCM texture features

### Lung Cancer Dataset

All plots are saved in `src/analysis/plots/`:

- `lung_class_distribution.png`: Bar plot showing the distribution of images across three classes
- `lung_intensity_distribution.png`: RGB channel intensity distributions for each class
- `lung_quality_metrics.png`: Distribution of blur, contrast, and brightness metrics
- `lung_texture_features.png`: Distribution of GLCM texture features
- `lung_sample_images.png`: Sample images from each class

## Analysis Features

### Image Quality Analysis

- Blur detection using Laplacian variance
- Contrast measurement
- Brightness analysis
- Aspect ratio statistics
- File size distribution

### Texture Analysis

- GLCM (Gray Level Co-occurrence Matrix) features:
  - Contrast
  - Dissimilarity
  - Homogeneity
  - Energy
  - Correlation
- Local Binary Patterns (LBP) features implementation available

### Color Analysis

- RGB channel statistics
- HSV color space analysis
- Mean and standard deviation for each channel
- Intensity distributions
