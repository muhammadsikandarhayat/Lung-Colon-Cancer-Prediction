import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis

# Create plots directory if it doesn't exist
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def get_image_stats(image_path):
    """Get basic statistics from an image"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Basic RGB stats
    basic_stats = {
        'mean_rgb': np.mean(img_rgb, axis=(0, 1)),
        'std_rgb': np.std(img_rgb, axis=(0, 1)),
        'min_rgb': np.min(img_rgb, axis=(0, 1)),
        'max_rgb': np.max(img_rgb, axis=(0, 1)),
        'shape': img.shape,
        'aspect_ratio': img.shape[1] / img.shape[0],
        'file_size': os.path.getsize(image_path) / 1024  # KB
    }
    
    # Color space statistics
    basic_stats.update({
        'mean_hsv': np.mean(img_hsv, axis=(0, 1)),
        'std_hsv': np.std(img_hsv, axis=(0, 1))
    })
    
    # Image quality metrics
    basic_stats['blur_score'] = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    basic_stats['contrast'] = img_gray.std()
    basic_stats['brightness'] = img_gray.mean()
    
    return basic_stats

def get_texture_features(img_gray):
    """Extract texture features using GLCM"""
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0]
    }

def get_lbp_features(img_gray):
    """Extract Local Binary Pattern features"""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def analyze_dataset():
    # Define paths
    base_path = Path("../../dataset/lung_image_sets")
    classes = ['lung_aca', 'lung_n', 'lung_scc']
    
    # Collect dataset statistics
    dataset_stats = {
        'class_distribution': {},
        'image_dimensions': set(),
        'mean_intensities': [],
        'class_samples': [],
        'quality_metrics': [],
        'texture_features': [],
        'aspect_ratios': [],
        'file_sizes': [],
        'sample_images': {}  # Store sample images for visualization
    }
    
    for class_name in classes:
        class_path = base_path / class_name
        images = list(class_path.glob('*.jpeg'))
        dataset_stats['class_distribution'][class_name] = len(images)
        
        print(f"\nAnalyzing {class_name} images...")
        # Store first image for sample visualization
        sample_img = cv2.imread(str(images[0]))
        dataset_stats['sample_images'][class_name] = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        
        for img_path in images[:20]:  # Analyze first 20 images as sample
            # Basic stats
            stats = get_image_stats(str(img_path))
            dataset_stats['image_dimensions'].add(str(stats['shape']))
            dataset_stats['mean_intensities'].append(stats['mean_rgb'])
            dataset_stats['class_samples'].append(class_name)
            dataset_stats['aspect_ratios'].append(stats['aspect_ratio'])
            dataset_stats['file_sizes'].append(stats['file_size'])
            
            # Quality metrics
            dataset_stats['quality_metrics'].append({
                'class': class_name,
                'blur_score': stats['blur_score'],
                'contrast': stats['contrast'],
                'brightness': stats['brightness']
            })
            
            # Texture analysis
            img_gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
            texture_features = get_texture_features(img_gray)
            texture_features['class'] = class_name
            dataset_stats['texture_features'].append(texture_features)
    
    return dataset_stats

def plot_class_distribution(stats):
    plt.figure(figsize=(12, 6))
    plt.bar(stats['class_distribution'].keys(), stats['class_distribution'].values())
    plt.title('Class Distribution in Lung Cancer Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'lung_class_distribution.png')
    plt.close()

def plot_intensity_distribution(stats):
    df = pd.DataFrame(stats['mean_intensities'], columns=['R', 'G', 'B'])
    df['Class'] = stats['class_samples']
    
    plt.figure(figsize=(15, 5))
    for i, channel in enumerate(['R', 'G', 'B']):
        plt.subplot(1, 3, i+1)
        for class_name in df['Class'].unique():
            class_data = df[df['Class'] == class_name][channel]
            plt.hist(class_data, alpha=0.5, label=class_name, bins=20)
        plt.title(f'{channel} Channel Distribution')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'lung_intensity_distribution.png')
    plt.close()

def plot_quality_metrics(stats):
    df = pd.DataFrame(stats['quality_metrics'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['blur_score', 'contrast', 'brightness']
    
    for i, metric in enumerate(metrics):
        for class_name in df['class'].unique():
            class_data = df[df['class'] == class_name][metric]
            axes[i].hist(class_data, alpha=0.5, label=class_name, bins=20)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'lung_quality_metrics.png')
    plt.close()

def plot_texture_features(stats):
    df = pd.DataFrame(stats['texture_features'])
    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        for class_name in df['class'].unique():
            class_data = df[df['class'] == class_name][feature]
            axes[i].hist(class_data, alpha=0.5, label=class_name, bins=20)
        axes[i].set_title(f'{feature.title()} Distribution')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'lung_texture_features.png')
    plt.close()

def plot_sample_images(stats):
    plt.figure(figsize=(15, 5))
    for i, (class_name, img) in enumerate(stats['sample_images'].items()):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(f'Sample {class_name}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'lung_sample_images.png')
    plt.close()

if __name__ == "__main__":
    print("Starting Lung Cancer Dataset Analysis...")
    stats = analyze_dataset()
    
    print("\nDataset Statistics:")
    print(f"Number of classes: {len(stats['class_distribution'])}")
    print("\nClass distribution:")
    for class_name, count in stats['class_distribution'].items():
        print(f"{class_name}: {count} images")
    
    print("\nUnique image dimensions found:", stats['image_dimensions'])
    
    # Calculate and print additional statistics
    print("\nAspect Ratio Statistics:")
    print(f"Mean: {np.mean(stats['aspect_ratios']):.2f}")
    print(f"Std: {np.std(stats['aspect_ratios']):.2f}")
    
    print("\nFile Size Statistics (KB):")
    print(f"Mean: {np.mean(stats['file_sizes']):.2f}")
    print(f"Std: {np.std(stats['file_sizes']):.2f}")
    print(f"Min: {np.min(stats['file_sizes']):.2f}")
    print(f"Max: {np.max(stats['file_sizes']):.2f}")
    
    # Generate plots
    plot_class_distribution(stats)
    plot_intensity_distribution(stats)
    plot_quality_metrics(stats)
    plot_texture_features(stats)
    plot_sample_images(stats)
    
    print("\nAnalysis complete! Check the plots directory for visualizations.") 