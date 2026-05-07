# Api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Import the Hugging Face pipeline
from transformers import pipeline

app = FastAPI(title="Skin Detection API")

# Load the zero-shot image classification pipeline (downloads ~600MB model on first run)
# We use CLIP because it understands text prompts combined with images.
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

@app.get("/")
def home():
    return {"message": "Skin Detection API is running"}

@app.post("/check-skin/")
async def check_skin(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "File must be an image"})

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ---------------------------------------------------------
        # THE ZERO-SHOT GATEKEEPER
        # We define the categories we want the AI to choose from.
        # It will assign a percentage to each label.
        # ---------------------------------------------------------
        candidate_labels = [
            "human skin close-up", 
            "hairy human skin",
            "dark human skin",
            "light human skin",
            "skin with a rash or lesion",
            "beige clothing or fabric", 
            "wood texture or furniture", 
            "sand or nature background"
        ]

        # Run the image through the classifier
        results = classifier(image, candidate_labels=candidate_labels)

        # The results come back as a list of dictionaries sorted by highest score
        top_prediction = results[0]['label']
        top_score = results[0]['score']

        # Determine if the top prediction belongs to the "skin" categories
        skin_labels = [
            "human skin close-up", 
            "hairy human skin", 
            "dark human skin", 
            "light human skin",
            "skin with a rash or lesion"
        ]
        
        is_skin = top_prediction in skin_labels

        return {
            "success": True,
            "is_skin_image": is_skin,
            "confidence_score": round(top_score, 4),
            "detected_category": top_prediction,
            "all_predictions": {res['label']: round(res['score'], 3) for res in results},
            "message": "Passed to main detection model" if is_skin else "Rejected: Image is not skin"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})