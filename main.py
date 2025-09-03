from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import gdown  # for downloading the model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Model Setup ----------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")

# Create model folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("📥 Model not found. Downloading...")
    url = "https://drive.google.com/uc?id=1L84L6Wiy9_SCLjgdPvnBgoQH8VRCsG4v"  # <-- replace with your Google Drive file ID
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ---------------- Classes ----------------
classes = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate"
]

category_map = {
    "im_Dyskeratotic": "High Risk / Cancerous",
    "im_Koilocytotic": "Pre-cancerous",
    "im_Metaplastic": "Pre-cancerous",
    "im_Parabasal": "Normal",
    "im_Superficial-Intermediate": "Normal"
}

# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check model file."}
    try:
        img = Image.open(file.file).resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(arr)[0]

        result = dict(zip(classes, preds.tolist()))
        predicted_class = classes[np.argmax(preds)]
        predicted_category = category_map[predicted_class]

        # Add mapped details
        mapped_details = {
            cls: {
                "category": category_map.get(cls, "Unknown"),
                "confidence": float(score)
            }
            for cls, score in result.items()
        }

        return {
            "prediction": predicted_category,
            "confidence": float(np.max(preds)),
            "details": result,
            "mapped_details": mapped_details
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ---------------- Report Generation Endpoint ----------------
@app.post("/generate-report")
async def generate_report(
    prediction: str = Body(...),
    confidence: float = Body(...),
    details: dict = Body(...),
    patientName: str = Body(...),
    patientAge: int = Body(...),
    patientLocation: str = Body(...)
):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        c = canvas.Canvas(temp_file.name, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, 770, "🏥 Cervical Cancer Detection Report")
        c.setFont("Helvetica", 12)
        c.drawCentredString(width / 2, 750, "AI-Assisted Medical Report")

        # Patient Info
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 720, "Patient Information:")
        c.setFont("Helvetica", 12)
        c.drawString(70, 700, f"Name: {patientName}")
        c.drawString(70, 680, f"Age: {patientAge}")
        c.drawString(70, 660, f"Location: {patientLocation}")

        # Prediction Info
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 630, "Prediction Result:")
        c.setFont("Helvetica", 12)
        c.drawString(70, 610, f"Predicted Category: {prediction}")
        c.drawString(70, 590, f"Confidence: {confidence*100:.2f}%")

        # Confidence Breakdown
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 560, "Confidence Breakdown per Class:")
        y = 540
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y, "Class")
        c.drawString(250, y, "Category")
        c.drawString(400, y, "Confidence")
        c.line(50, y-2, 500, y-2)
        y -= 20
        c.setFont("Helvetica", 12)
        for cls, score in details.items():
            mapped_cls_category = category_map.get(cls, "Unknown")
            c.drawString(60, y, cls)
            c.drawString(250, y, mapped_cls_category)
            c.drawString(400, y, f"{score*100:.2f}%")
            y -= 20

        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 100, "⚕️ Disclaimer: AI-assisted report, not a medical diagnosis.")
        c.drawString(50, 85, "Consult a certified doctor for professional advice.")

        c.showPage()
        c.save()

        return FileResponse(temp_file.name, filename="Cancer_Report.pdf", media_type="application/pdf")
    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}
