import os
import json
import asyncio
import numpy as np # NEW

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Suppress TensorFlow logging spam in the console
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
from transformers import pipeline
from openai import AsyncOpenAI
from tensorflow.keras.models import load_model # NEW

app = FastAPI(title="Skin Detection API")

# ---------------------------------------------------------
# INITIALIZE MODELS
# ---------------------------------------------------------
# 1. CLIP Profiler
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# 2. Custom Disease Model (Model 2)
# Ensure "skin_disease_3class_model.h5" is in the same folder as this Api.py file
try:
    disease_model = load_model("skin_disease_3class_model.h5")
    # ---> UPDATE THESE WITH YOUR ACTUAL 3 CLASS NAMES <---
    DISEASE_CLASSES = ["Melanoma", "Basal Cell Carcinoma", "Melanocytic Nevi"]
except Exception as e:
    print(f"Warning: Could not load the disease model. Error: {e}")
    disease_model = None

# 3. DeepSeek LLM
DEEPSEEK_API_KEY = "sk-53eb8f65c2814dd7805a8f4dd433840f"
llm_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


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

        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        # --- STAGE 1: THE GATEKEEPER ---
        validity_labels = [
            "human skin, a human face, or a body part",
            "clothing, fabric, or textiles",
            "furniture, wood, or inanimate objects",
            "nature, outdoor landscape, or background",
        ]
        val_results = classifier(image, candidate_labels=validity_labels)
        
        if val_results[0]["label"] != "human skin, a human face, or a body part":
            return StreamingResponse(
                generate_llm_stream(is_valid=False, profile_data=None), 
                media_type="text/event-stream",
                headers=stream_headers
            )

        # --- STAGE 2: THE PROFILER ---
        tone_labels = ["very light pale skin tone", "light beige skin tone", "medium tan skin tone", "olive brown skin tone", "dark brown skin tone", "very dark brown skin tone"]
        hair_labels = ["skin with visible thick body hair", "bare skin without visible thick hair"]
        age_labels = ["older skin with deep aging wrinkles", "youthful skin without aging wrinkles"]
        health_labels = ["healthy clear flawless skin", "textured skin with acne, rash, blemishes, or disease"]

        tone_res = classifier(image, candidate_labels=tone_labels)
        hair_res = classifier(image, candidate_labels=hair_labels)
        age_res = classifier(image, candidate_labels=age_labels)
        health_res = classifier(image, candidate_labels=health_labels)
        
        health_status = health_res[0]["label"]

        profile_data = {
            "skin_tone": tone_res[0]["label"],
            "hair_presence": hair_res[0]["label"],
            "texture_and_age": age_res[0]["label"],
            "health_status": health_status,
            "confidence_scores": {
                "skin_tone": round(tone_res[0]["score"], 3),
                "hair_presence": round(hair_res[0]["score"], 3),
                "texture_and_age": round(age_res[0]["score"], 3),
                "health_status": round(health_res[0]["score"], 3),
            },
        }

        # --- STAGE 3: THE DISEASE SPECIALIST (CONDITIONAL ROUTING) ---
        # Only run the heavy .h5 model if the CLIP model flagged the skin as having a disease/texture
        if disease_model is not None and "textured skin" in health_status:
            # Resize image to what the model expects (usually 224x224 for TF models)
            # IF YOUR MODEL EXPECTS A DIFFERENT SIZE, CHANGE 224 to your size!
            img_resized = image.resize((224, 224))
            
            # Convert to numpy array and normalize (0-255 -> 0.0-1.0)
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            
            # Predict
            predictions = disease_model.predict(img_array)[0]
            top_index = np.argmax(predictions)
            top_class = DISEASE_CLASSES[top_index]
            top_confidence = round(float(predictions[top_index]), 3)
            
            # Append to our JSON profile
            profile_data["disease_prediction"] = {
                "detected_class": top_class,
                "confidence_score": top_confidence
            }

        # --- STAGE 4: THE DEEPSEEK LLM ---
        return StreamingResponse(
            generate_llm_stream(is_valid=True, profile_data=profile_data), 
            media_type="text/event-stream",
            headers=stream_headers
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


async def generate_llm_stream(is_valid: bool, profile_data: dict):
    initial_data = {
        "type": "metadata",
        "is_skin": is_valid,
        "profile": profile_data
    }
    yield f"data: {json.dumps(initial_data)}\n\n"

    if not is_valid:
        system_prompt = "You are Skinalyze. The uploaded image is NOT human skin. Keep your response under 2 sentences. Politely ask the user to upload a clear image of their skin."
        user_prompt = "Generate the rejection response."
    else:
        # We update the LLM prompt to look for the new "disease_prediction" key
        system_prompt = """
        You are Skinalyze, a professional AI dermatology assistant. 
        Analyze the visual profile.
        
        CRITICAL FORMATTING RULES:
        1. Be highly concise. Use Markdown (bullet points, bolding).
        2. Visual Profile: Bullet point the Skin Tone, Hair Presence, and Texture.
        3. Medical Analysis: 
            - If 'disease_prediction' is present in the data, create a specific heading called "Initial Screening". State the detected condition and confidence. MUST include a strict disclaimer that you are an AI and they must consult a dermatologist.
            - If 'disease_prediction' is NOT in the data, state that the skin appears clear and healthy.
        4. Do not write long paragraphs. Keep it punchy for mobile.
        """
        user_prompt = f"Here is the data: {profile_data}. Generate the analysis."

    try:
        response = await llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
            stream=True, 
        )

        async for chunk in response:
            text_chunk = chunk.choices[0].delta.content
            if text_chunk:
                chunk_data = {"type": "text", "content": text_chunk}
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.03)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"