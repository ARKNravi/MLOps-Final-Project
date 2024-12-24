from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from train import initialize_model
import os
from supabase import create_client
import uuid
from datetime import datetime

app = FastAPI(title="Coffee Bean Classifier API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Model setup
device = torch.device("cpu")
model = initialize_model(4).to(device)

# Load model
def load_model():
    try:
        model_path = os.path.join("model", "best_model_weights.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Transform for images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and transform image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get class name
        classes = ['defect', 'longberry', 'peaberry', 'premium']
        prediction = classes[predicted.item()]
        
        # Create response
        response = {
            "prediction": prediction,
            "confidence": float(probabilities[predicted.item()]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to Supabase
        supabase.table("predictions").insert(response).execute()
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Coffee Bean Classifier API is running"}

# Load model on startup
load_model()