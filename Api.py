import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from transformers import pipeline

app = FastAPI(title="Skin Detection API")

# Load the zero-shot image classification pipeline
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
        # STAGE 1: THE GATEKEEPER (Skin vs. Non-Skin)
        # ---------------------------------------------------------
        # FIX: Expanded the prompt to accept faces and general body parts
        # so it doesn't reject clear portraits.
        validity_labels = [
            "human skin, a human face, or a body part", 
            "clothing, fabric, or textiles", 
            "furniture, wood, or inanimate objects", 
            "nature, outdoor landscape, or background"
        ]
        
        val_results = classifier(image, candidate_labels=validity_labels)
        top_validity = val_results[0]['label']

        if top_validity != "human skin, a human face, or a body part":
            return {
                "success": False,
                "is_skin_image": False,
                "message": "Rejected: Please upload a clear image of human skin or a face."
            }

# ---------------------------------------------------------
        # STAGE 2: THE PROFILER (Analyzing the valid skin)
        # ---------------------------------------------------------
        
        # FIX: Purely color-based descriptions. Removed "pigmented" to stop 
        # the AI from confusing acne marks with dark skin tones.
        tone_labels = [
            "very light pale skin tone",
            "light beige skin tone",
            "medium tan skin tone",
            "olive brown skin tone",
            "dark brown skin tone",
            "very dark brown skin tone"
        ]
        
        # FIX: Added "thick" to stop it from flagging tiny peach-fuzz or pores as hair
        hair_labels = [
            "skin with visible thick body hair", 
            "bare skin without visible thick hair"
        ]
        
        # FIX: Specified "aging wrinkles" so it stops confusing acne bumps with old age
        age_labels = [
            "older skin with deep aging wrinkles", 
            "youthful skin without aging wrinkles"
        ]
        
        # FIX: Grouped all textures and blemishes here so they don't bleed into the other categories
        health_labels = [
            "healthy clear flawless skin", 
            "textured skin with acne, rash, blemishes, or disease"
        ]

        tone_res = classifier(image, candidate_labels=tone_labels)
        hair_res = classifier(image, candidate_labels=hair_labels)
        age_res = classifier(image, candidate_labels=age_labels)
        health_res = classifier(image, candidate_labels=health_labels)

        # ---------------------------------------------------------
        # FINAL RESPONSE
        # ---------------------------------------------------------
        return {
            "success": True,
            "is_skin_image": True,
            "profile": {
                "skin_tone": tone_res[0]['label'],
                "hair_presence": hair_res[0]['label'],
                "texture_and_age": age_res[0]['label'],
                "health_status": health_res[0]['label']
            },
            "confidence_scores": {
                "skin_tone": round(tone_res[0]['score'], 3),
                "hair_presence": round(hair_res[0]['score'], 3),
                "texture_and_age": round(age_res[0]['score'], 3),
                "health_status": round(health_res[0]['score'], 3)
            },
            "message": "Valid skin image. Detailed profile generated."
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})