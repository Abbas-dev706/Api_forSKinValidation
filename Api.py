import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Configuration ---
# Use the best model saved during your training!
MODEL_PATH = "best_skin_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MUST match the exact order of your training script folders
CLASS_NAMES = ['Basal Cell Carcinoma (BCC)', 'Melanocytic Nevi (NV)', 'Melanoma', 'normal']

# --- Initialize App ---
app = FastAPI(title="Skin Disease Detection API", version="1.0")

# Allow mobile apps or web frontends to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load Model ---
def load_model():
    print(f"Loading model on {DEVICE}...")
    model = models.resnet50(weights=None)  # We don't need pretrained weights now, we have our own!
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 classes

    # Load your trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    return model


model = load_model()

# --- Image Preprocessing ---
# This MUST be identical to your val_transforms from training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Skin Disease Classification API is running!"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 1. Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # 2. Preprocess the image
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # 3. Make Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        # 4. Format the result
        prediction_label = CLASS_NAMES[predicted_idx.item()]
        confidence_score = float(confidence.item()) * 100

        # All probabilities for debugging/charting in your app
        all_probs = {CLASS_NAMES[i]: float(probabilities[i].item() * 100) for i in range(4)}

        return {
            "status": "success",
            "prediction": prediction_label,
            "confidence": f"{confidence_score:.2f}%",
            "details": all_probs
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)