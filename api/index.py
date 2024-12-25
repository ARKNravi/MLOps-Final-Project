from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import io
import os
from pathlib import Path

app = FastAPI(title="Coffee Bean Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi model dan transformasi
try:
    device = torch.device("cpu")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 4)  # 4 kelas

    # Load model weights
    current_dir = Path(__file__).parent
    model_path = os.path.join(current_dir, "model", "best_model_weights.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Kelas biji kopi
class_names = ['Defect', 'Longberry', 'Peaberry', 'Premium']

@app.get("/")
async def read_root():
    return {
        "message": "Selamat datang di API Klasifikasi Biji Kopi",
        "model_status": "loaded" if model is not None else "not loaded",
        "endpoints": {
            "root": "/",
            "test": "/api/test",
            "predict": "/api/predict (POST, membutuhkan file gambar)"
        }
    }

@app.get("/api/test")
async def test():
    try:
        if model is None:
            return {
                "status": "error",
                "message": "Model belum dimuat"
            }
        return {
            "status": "success",
            "message": "API is working and model is loaded!"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {
            "success": False,
            "error": "Model not loaded"
        }
        
    try:
        # Baca gambar
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Konversi ke RGB jika dalam mode lain
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Transformasi gambar
        image_tensor = transform(image).unsqueeze(0)
        
        # Prediksi
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Hitung probabilitas dengan softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Ambil probabilitas untuk setiap kelas
            class_probs = {class_names[i]: float(probabilities[i]) * 100 for i in range(len(class_names))}
            
            # Ambil prediksi dengan probabilitas tertinggi
            predicted_class = class_names[predicted.item()]
            confidence = float(probabilities[predicted.item()]) * 100
            
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "class_probabilities": {k: f"{v:.2f}%" for k, v in class_probs.items()}
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }