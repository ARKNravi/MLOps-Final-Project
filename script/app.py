from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os
from pathlib import Path

# Debug statements
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Files in script directory:", os.listdir(Path(__file__).parent))

# Tambahkan script directory ke Python path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
    print("Added to path:", str(script_dir))

try:
    from model_utils import initialize_model
    print("Successfully imported initialize_model")
except Exception as e:
    print("Error importing initialize_model:", str(e))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi model
device = torch.device("cpu")
num_classes = 4
model = initialize_model(num_classes)

# Load model weights
model.load_state_dict(torch.load("model/best_model_weights.pth", map_location=device))
model.eval()

# Definisikan transformasi gambar
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Definisikan kelas
class_names = ['Black Bean', 'Brown Bean', 'Green Bean', 'White Bean']

@app.get("/")
def read_root():
    return {"message": "Selamat datang di API Klasifikasi Biji Kopi"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
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
            class_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            
            # Ambil prediksi dengan probabilitas tertinggi
            predicted_class = class_names[predicted.item()]
            confidence = float(probabilities[predicted.item()])
            
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs
        }
        
    except Exception as e:
        return {"error": str(e)}