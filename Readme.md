# Lung-Colon Cancer Prediction AI

This project implements a deep learning system for detecting lung and colon cancer from medical images using convolutional neural networks (CNNs). The system includes both training scripts and a web application for real-time inference.

## Features

- **Deep Learning Models**: Custom CNN architecture for cancer detection
- **Transfer Learning**: Support for ResNet and EfficientNet models
- **Web Application**: FastAPI backend with React frontend for easy testing
- **Comprehensive Analysis**: EDA scripts and model comparison tools
- **Multiple Cancer Types**: Support for both lung and colon cancer detection

## Project Structure

```
Lung-Colon-Cancer-Prediction/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API endpoints and server
├── frontend/               # React frontend
│   ├── public/
│   ├── src/
│   └── package.json
├── src/                    # Core ML code
│   ├── models/            # Model definitions and training
│   ├── analysis/          # EDA and analysis scripts
│   └── compare_models.py  # Model comparison
├── dataset/               # Dataset storage
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Backend Setup

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI backend:**

   ```bash
   cd backend
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### 2. Frontend Setup

1. **Install Node.js dependencies:**

   ```bash
   cd frontend
   npm install
   ```

2. **Start the React development server:**

   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000` port configured in vite.config.js

## API Endpoints

- `GET /models` - Get info about available models
- `POST /predict/{cancer_type}` - Predict cancer from uploaded image
  - `cancer_type`: Either 'lung' or 'colon'
  - `model_type`: Either 'cnn', 'resnet', 'efficientnet' or all
  - `file`: Image file (JPEG, PNG, etc.)

## Frontend Features

- **Step-by-Step Interface**: Guided workflow for cancer type and model selection
- **Modern UI**: Clean, responsive design with gradient backgrounds and step indicators
- **Image Upload**: Drag-and-drop or click-to-upload functionality
- **Real-time Preview**: Image preview before prediction
- **Cancer Type Selection**: Choose between lung and colon cancer models
- **Model Selection**: Choose between custom CNN, transfer learning models (ResNet, EfficientNet), or all models for comparison
- **Multi-Model Comparison**: Compare predictions from all three models simultaneously
- **Detailed Results**: Confidence scores and probability breakdowns
- **Visual Feedback**: Color-coded results and loading states

## Model Information

### CNN Architecture

- 3 convolutional layers with batch normalization
- Max pooling for dimensionality reduction
- Dropout for regularization
- 2-class classification (Normal vs Cancer)

### Available Models

**CNN Models (Custom Architecture):**

- **Lung Cancer Model**: `src/models/cnn/checkpoints/lung_model.pth`
- **Colon Cancer Model**: `src/models/cnn/checkpoints/colon_model.pth`

**ResNet Models (Transfer Learning):**

- **Lung Cancer Model**: `src/models/transfer_learning/ResNet/checkpoints/lung_resnet.pth`
- **Colon Cancer Model**: `src/models/transfer_learning/ResNet/checkpoints/colon_resnet.pth`

**EfficientNet Models (Transfer Learning):**

- **Lung Cancer Model**: `src/models/transfer_learning/EfficientNet/checkpoints/lung_efficientnet.pth`
- **Colon Cancer Model**: `src/models/transfer_learning/EfficientNet/checkpoints/colon_efficientnet.pth`

## Training

To train new models or retrain existing ones:

```bash
# Train CNN models
cd src/models/cnn
python train.py

# Train transfer learning models
cd src/models/transfer_learning/ResNet
python train_resnet.py

cd src/models/transfer_learning/EfficientNet
python train_efficientnet.py
```

## Analysis

Run exploratory data analysis:

```bash
cd src/analysis
python eda_lungs.py
python eda_colon.py
```

Compare model performances:

```bash
cd src
python compare_models.py
```

## Requirements

### Python Dependencies

- FastAPI
- PyTorch
- Torchvision
- PIL (Pillow)
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Node.js Dependencies

- React 18
- Axios
- Lucide React (icons)

## Usage

1. Start both backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Select cancer type (lung or colon)
4. Choose a specific model
5. Upload a medical image
6. Click "Predict Cancer" to get results
7. View detailed prediction(s) with confidence scores and probability breakdowns

## Important Notes

- This is a research tool and should not be used as a substitute for professional medical diagnosis
- Always consult with healthcare professionals for medical decisions
- The models are trained on specific datasets and may not generalize to all medical images
- Ensure proper data privacy and security when handling medical images

## License

This project is for research and educational purposes only.


