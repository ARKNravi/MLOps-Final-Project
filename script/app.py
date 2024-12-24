import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
from train import initialize_model
import os
from supabase import create_client, Client
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Optional
import sys

# Inisialisasi FastAPI
api = FastAPI(title="Coffee Bean Classifier API")

# Konfigurasi CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model untuk feedback
class FeedbackModel(BaseModel):
    prediction_id: str
    predicted_class: str
    actual_class: str
    is_correct: bool

# Inisialisasi Supabase
supabase: Client = create_client(
    supabase_url=st.secrets["supabase_url"],
    supabase_key=st.secrets["supabase_key"]
)

# Konfigurasi model
device = torch.device("cpu")
num_classes = 4
model = initialize_model(num_classes).to(device)

# Load model weights
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best_model_weights.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model
        else:
            st.error(f"Model weights tidak ditemukan di path: {model_path}")
            st.info("Pastikan file model tersedia di direktori yang benar.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Coba restart aplikasi jika error berlanjut.")
        return None

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Kelas biji kopi
class_names = ['Arabica', 'Liberica', 'Robusta', 'Mixed']

def predict_image(image):
    try:
        # Transformasi gambar
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prediksi
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # Probabilitas untuk setiap kelas
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        return {
            'class': class_names[predicted.item()],
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(class_names, probabilities)
            }
        }
    except Exception as e:
        return {'error': str(e)}

def save_feedback(prediction_id, image_url, predicted_class, actual_class, is_correct):
    try:
        feedback_data = {
            'prediction_id': prediction_id,
            'image_url': image_url,
            'predicted_class': predicted_class,
            'actual_class': actual_class,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        result = supabase.table('feedback').insert(feedback_data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

# API Endpoints
@api.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validasi file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
        # Baca gambar
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prediksi
        result = predict_image(image)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        return {
            "prediction_id": prediction_id,
            "prediction": result['class'],
            "probabilities": result['probabilities']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/feedback")
async def feedback(feedback_data: FeedbackModel):
    try:
        success = save_feedback(
            prediction_id=feedback_data.prediction_id,
            image_url="",
            predicted_class=feedback_data.predicted_class,
            actual_class=feedback_data.actual_class,
            is_correct=feedback_data.is_correct
        )
        
        if success:
            return {"status": "success", "message": "Feedback recorded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save feedback")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI
def main():
    st.title("☕ Klasifikasi Biji Kopi")

    # Load model
    model = load_model()

    # Sidebar
    st.sidebar.title("Tentang")
    st.sidebar.info(
        "Aplikasi ini menggunakan deep learning untuk mengklasifikasikan "
        "jenis biji kopi berdasarkan gambar. Upload gambar biji kopi "
        "dan dapatkan prediksi jenisnya!"
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload gambar biji kopi...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diupload', use_column_width=True)
        
        # Prediksi
        if model is not None:
            with st.spinner('Menganalisis gambar...'):
                result = predict_image(image)
                
            if 'error' not in result:
                # Tampilkan hasil
                st.success(f"Prediksi: {result['class']}")
                
                # Tampilkan probabilitas
                st.subheader("Probabilitas per Kelas:")
                for class_name, prob in result['probabilities'].items():
                    st.progress(prob)
                    st.write(f"{class_name}: {prob:.2%}")
                
                # Feedback
                st.subheader("Feedback")
                is_correct = st.radio("Apakah prediksi ini benar?", ("Ya", "Tidak"))
                
                if is_correct == "Tidak":
                    actual_class = st.selectbox("Pilih kelas yang benar:", class_names)
                else:
                    actual_class = result['class']
                
                if st.button("Kirim Feedback"):
                    # Generate unique ID untuk prediksi
                    prediction_id = str(uuid.uuid4())
                    
                    # Simpan feedback
                    if save_feedback(
                        prediction_id=prediction_id,
                        image_url="",
                        predicted_class=result['class'],
                        actual_class=actual_class,
                        is_correct=(is_correct == "Ya")
                    ):
                        st.success("Terima kasih atas feedback Anda!")
                    else:
                        st.error("Gagal menyimpan feedback.")
            else:
                st.error(f"Error dalam prediksi: {result['error']}")

    # API Documentation
    st.sidebar.title("API Documentation")
    if st.sidebar.checkbox("Show API Documentation"):
        st.subheader("API Documentation")
        st.markdown("""
        ### POST /predict
        
        Endpoint untuk prediksi gambar biji kopi.
        
        **Request Body (form-data):**
        - `file`: File gambar (jpg/jpeg/png)
        
        **Response:**
        ```json
        {
            "prediction_id": "uuid",
            "prediction": "Nama Kelas",
            "probabilities": {
                "Arabica": 0.xx,
                "Liberica": 0.xx,
                "Robusta": 0.xx,
                "Mixed": 0.xx
            }
        }
        ```
        
        ### POST /feedback
        
        Endpoint untuk memberikan feedback hasil prediksi.
        
        **Request Body (JSON):**
        ```json
        {
            "prediction_id": "uuid",
            "predicted_class": "Nama Kelas",
            "actual_class": "Nama Kelas",
            "is_correct": boolean
        }
        ```
        
        **Response:**
        ```json
        {
            "status": "success",
            "message": "Feedback recorded"
        }
        ```
        """)

if __name__ == "__main__":
    try:
        # Setup logging
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Log environment info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Torch version: {torch.__version__}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check environment variables
        if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
            st.error("Error: Missing required environment variables. Please check your deployment configuration.")
            logger.error("Missing required environment variables")
        else:
            # Initialize model
            logger.info("Initializing application...")
            main()
            
            # Run FastAPI if needed
            if os.environ.get("FASTAPI_SERVER", "false").lower() == "true":
                logger.info("Starting FastAPI server...")
                uvicorn.run(api, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Error: {str(e)}")
        st.info("If you're seeing this error, please check the logs for more details.") 