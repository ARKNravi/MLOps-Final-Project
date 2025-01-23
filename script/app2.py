import os
import streamlit as st
import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Coffee Bean Quality Classification",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        /* Main title styling */
        .stTitle {
            font-size: 42px;
            text-align: center;
            padding: 30px 0;
            color: #FFFFFF;
            font-weight: 600;
        }
        
        /* Subtitle styling */
        .subtitle {
            font-size: 24px;
            text-align: center;
            color: #E0E0E0;
            margin-bottom: 40px;
        }
        
        /* Card container styling */
        .card {
            background-color: #2D2D2D;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
        }
        
        /* Upload section styling */
        .upload-section {
            text-align: center;
            padding: 20px;
            border: 2px dashed #4A4A4A;
            border-radius: 15px;
            margin: 20px 0;
            background-color: #2D2D2D;
        }
        
        /* Result container styling */
        .result-container {
            background-color: #2D2D2D;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        /* Confidence meter styling */
        .confidence-meter {
            background-color: #3D3D3D;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        /* Progress bar container */
        .progress-bar-container {
            background-color: #2D2D2D;
            border-radius: 10px;
            margin-top: 8px;
            position: relative;
            height: 20px;
        }
        
        /* Progress bar styling */
        .progress-bar {
            height: 20px;
            border-radius: 10px;
            transition: width 0.5s ease-in-out;
            position: relative;
        }
        
        /* Progress bar text */
        .progress-bar-text {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            color: #FFFFFF;
            font-weight: 500;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
            z-index: 1;
        }
        
        /* Classification result styling */
        .classification-result {
            font-size: 28px;
            color: #FFFFFF;
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            background-color: #3D3D3D;
            border-radius: 10px;
            font-weight: 600;
        }
        
        /* Description text styling */
        .description-text {
            color: #E0E0E0;
            text-align: center;
            font-style: italic;
            margin: 15px 0;
            font-size: 16px;
        }
        
        /* Meter label styling */
        .meter-label {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: 500;
        }
        
        /* Meter value styling */
        .meter-value {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: 600;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 30px 0;
            color: #B0B0B0;
            border-top: 1px solid #3D3D3D;
            margin-top: 50px;
        }
        
        /* Headers */
        h1, h2, h3 {
            text-align: center;
            color: #FFFFFF;
        }
        
        /* Analysis confidence header */
        .analysis-header {
            color: #FFFFFF;
            font-size: 22px;
            text-align: center;
            margin: 25px 0;
            font-weight: 500;
        }
        
        /* Image container */
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        
        /* Center the image within the container */
        .image-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables and setup MLflow
load_dotenv()
MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')

# Model loading with caching
@st.cache_resource
def load_model():
    model_uri = "runs:/5f0339a0e0ff4c56bf4ac603bd8ccada/best_model"
    model = mlflow.pytorch.load_model(model_uri)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

model, device = load_model()
model.eval()

# Image transformation
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ["Defect", "Longberry", "Peaberry", "Premium"]
class_descriptions = {
    "Defect": "Beans with visible imperfections or damage that may affect coffee quality",
    "Longberry": "Distinguished by their elongated shape, typically producing a unique flavor profile",
    "Peaberry": "Natural mutation where the coffee cherry produces a single, round bean instead of two flat sides",
    "Premium": "Highest quality beans meeting strict grading standards for size, color, and density"
}

def predict(image: Image.Image):
    img_t = inference_transform(image)
    batch_t = img_t.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(batch_t)
        probs = F.softmax(outputs, dim=1)
        
    all_probs = probs[0].cpu().numpy()
    pred_prob, pred_class_idx = torch.max(probs, dim=1)
    
    return (
        class_names[pred_class_idx.item()],
        pred_prob.item(),
        {class_names[i]: float(prob) for i, prob in enumerate(all_probs)}
    )

# Main UI
st.markdown("<h1>Coffee Bean Quality Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered coffee bean analysis system</p>", unsafe_allow_html=True)

# Center container
container = st.container()
with container:
    left_col, main_col, right_col = st.columns([1, 2, 1])
    
    with main_col:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your coffee bean image", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", width=400)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner('Analyzing your coffee beans...'):
                pred_class_name, pred_prob, all_probs = predict(image)
                
                st.markdown(f'<div class="classification-result">{pred_class_name}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="description-text">{class_descriptions[pred_class_name]}</p>', unsafe_allow_html=True)
                
                st.markdown('<div class="analysis-header">Analysis Confidence</div>', unsafe_allow_html=True)
                
                for class_name, prob in all_probs.items():
                    progress_color = "#00B4D8" if class_name == pred_class_name else "#4A4A4A"
                    st.markdown(
                        f'<div class="confidence-meter">'
                        f'<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
                        f'<span class="meter-label">{class_name}</span>'
                        f'<span class="meter-value">{prob:.1%}</span>'
                        f'</div>'
                        f'<div class="progress-bar-container">'
                        f'<div class="progress-bar" style="width: {prob*100:.0f}%; background-color: {progress_color};">'
                        f'<span class="progress-bar-text">{prob:.1%}</span>'
                        f'</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>📸 For best results, use well-lit, clear images of coffee beans</p>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
""", unsafe_allow_html=True)