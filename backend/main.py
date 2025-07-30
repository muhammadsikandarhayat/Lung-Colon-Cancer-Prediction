from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
from pathlib import Path
import sys

# Add the src directory to the path to import our models
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.cnn.cnn_model import CancerCNN

app = FastAPI(title="Cancer Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models
loaded_models = {}
model_paths = {
    "lung_cnn": "../src/models/cnn/checkpoints/lung_model.pth",
    "colon_cnn": "../src/models/cnn/checkpoints/colon_model.pth",
    "lung_resnet": "../src/models/transfer_learning/ResNet/checkpoints/lung_resnet.pth",
    "colon_resnet": "../src/models/transfer_learning/ResNet/checkpoints/colon_resnet.pth",
    "lung_efficientnet": "../src/models/transfer_learning/EfficientNet/checkpoints/lung_efficientnet.pth",
    "colon_efficientnet": "../src/models/transfer_learning/EfficientNet/checkpoints/colon_efficientnet.pth"
}

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(cancer_type: str, model_type: str = "cnn"):
    """Load a trained model for the specified cancer type and model type"""
    global loaded_models
    
    valid_cancer_types = ["lung", "colon"]
    valid_model_types = ["cnn", "resnet", "efficientnet"]
    
    if cancer_type not in valid_cancer_types:
        raise ValueError(f"Invalid cancer type: {cancer_type}. Must be one of {valid_cancer_types}")
    
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_model_types}")
    
    model_key = f"{cancer_type}_{model_type}"
    
    if model_key not in loaded_models:
        model_path = model_paths[model_key]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize model based on type
        if model_type == "cnn":
            model = CancerCNN()
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            # Handle both dict format and direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        elif model_type == "resnet":
            from torchvision import models as torchvision_models
            model = torchvision_models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
        elif model_type == "efficientnet":
            from torchvision import models as torchvision_models
            model = torchvision_models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
        
        model.eval()
        loaded_models[model_key] = model
    
    return loaded_models[model_key]

def preprocess_image(image_bytes: bytes):
    """Preprocess uploaded image for model inference"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

@app.post("/predict/{cancer_type}")
async def predict_cancer(cancer_type: str, file: UploadFile = File(...), model_type: str = "cnn"):
    try:
        # Validate cancer type
        if cancer_type not in ["lung", "colon"]:
            raise HTTPException(status_code=400, detail="Invalid cancer type. Must be 'lung' or 'colon'")
        
        # Validate model type
        if model_type not in ["cnn", "resnet", "efficientnet", "all"]:
            raise HTTPException(status_code=400, detail="Invalid model type. Must be 'cnn', 'resnet', 'efficientnet', or 'all'")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Map class indices to labels based on training data structure
        if cancer_type == "lung":
            # Lung: 0=normal (lung_n), 1=cancer (lung_aca + lung_scc)
            class_labels = {0: "Normal Lung Tissue", 1: "Lung Cancer"}
        else:  # colon
            # Colon: 0=normal (colon_n), 1=cancer (colon_aca)
            class_labels = {0: "Normal Colon Tissue", 1: "Colon Cancer"}
        
        if model_type == "all":
            # Predict with all models
            results = []
            model_types = ["cnn", "resnet", "efficientnet"]
            
            for mt in model_types:
                try:
                    # Load model
                    model = load_model(cancer_type, mt)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    predicted_label = class_labels[predicted_class]
                    
                    # Get probabilities for both classes
                    normal_prob = probabilities[0][0].item()
                    cancer_prob = probabilities[0][1].item()
                    
                    results.append({
                        "model_type": mt,
                        "prediction": predicted_label,
                        "confidence": confidence,
                        "probabilities": {
                            "normal": normal_prob,
                            "cancer": cancer_prob
                        }
                    })
                except Exception as e:
                    results.append({
                        "model_type": mt,
                        "error": str(e)
                    })
            
            return {
                "cancer_type": cancer_type,
                "model_type": "all",
                "predictions": results,
                "filename": file.filename
            }
        else:
            model = load_model(cancer_type, model_type)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_label = class_labels[predicted_class]
            
            # Get probabilities for both classes
            normal_prob = probabilities[0][0].item()
            cancer_prob = probabilities[0][1].item()
            
            return {
                "cancer_type": cancer_type,
                "model_type": model_type,
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": {
                    "normal": normal_prob,
                    "cancer": cancer_prob
                },
                "filename": file.filename
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get information about available models"""
    available_models = {}
    
    for model_key, model_path in model_paths.items():
        cancer_type, model_type = model_key.split('_')
        if cancer_type not in available_models:
            available_models[cancer_type] = {}
        
        available_models[cancer_type][model_type] = {
            "path": model_path,
            "exists": os.path.exists(model_path),
            "loaded": model_key in loaded_models
        }
    
    return available_models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 