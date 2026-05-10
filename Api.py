import os
import json
import asyncio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- NEW: Tell Hugging Face to NEVER look for TensorFlow ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

# IMPORT Query so we can use URL parameters
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
from transformers import pipeline
from openai import AsyncOpenAI

# --- NEW: PyTorch Imports ---
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="Skin Detection API")

# ---------------------------------------------------------
# INITIALIZE MODELS
# ---------------------------------------------------------
classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    framework="pt",
)

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PyTorch model on {DEVICE}...")

    # Initialize ResNet50 for 4 classes
    disease_model = models.resnet50(weights=None)
    num_ftrs = disease_model.fc.in_features
    disease_model.fc = nn.Linear(num_ftrs, 4)

    # Load your trained PyTorch weights
    disease_model.load_state_dict(
        torch.load("skin_disease_final.pth", map_location=DEVICE)
    )
    disease_model.to(DEVICE)
    disease_model.eval()

    # MUST match your training folders exactly
    DISEASE_CLASSES = [
        "Basal Cell Carcinoma (BCC)",
        "Melanocytic Nevi (NV)",
        "Melanoma",
        "normal",
    ]

    # Image Preprocessing for PyTorch
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    print("PyTorch Model loaded successfully!")

except Exception as e:
    print(f"Warning: Could not load the disease model. Error: {e}")
    disease_model = None

DEEPSEEK_API_KEY = "sk-53eb8f65c2814dd7805a8f4dd433840f"
llm_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


@app.get("/")
def home():
    return {"message": "Skin Detection API is running"}


@app.post("/check-skin/")
async def check_skin(
    file: UploadFile = File(...),
    stream: bool = Query(
        True, description="Set to false to get standard JSON instead of streaming"
    ),
):
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400, content={"error": "File must be an image"}
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        # --- STAGE 1: THE UNIVERSAL GATEKEEPER ---
        TARGET_LABEL = (
            "a clear, close-up photograph of human skin, a human face, or a body part"
        )

        validity_labels = [
            TARGET_LABEL,
            "a blurry, completely out-of-focus, or unreadable image",
            "a photo of a computer screen, laptop monitor, or digital display",
            "a digital illustration, cartoon, anime, or 3D graphic",
            "a document, text, screenshot, receipt, or meme",
            "an animal, pet, dog, cat, or wildlife",
            "food, meals, groceries, or beverages",
            "a vehicle, car, motorcycle, or transportation",
            "a building, indoor room, architecture, or outdoor landscape",
            "clothing, fabric, shoes, or fashion accessories",
            "an inanimate object, tool, gadget, toy, or furniture",
            "a wide shot of a crowd or a group of people standing far away",
        ]

        val_results = classifier(image, candidate_labels=validity_labels)
        top_validity = val_results[0]["label"]

        print(f"\n--- GATEKEEPER SCORES ---")
        for i in range(3):
            print(f"{val_results[i]['label']}: {val_results[i]['score']:.3f}")
        print("-------------------------\n")

        if top_validity != TARGET_LABEL:
            if stream:
                return StreamingResponse(
                    generate_llm_stream(is_valid=False, profile_data=None),
                    media_type="text/event-stream",
                    headers=stream_headers,
                )
            else:
                return await generate_llm_json(is_valid=False, profile_data=None)

        # --- STAGE 2: THE COSMETIC & MINOR CONDITION PROFILER (CLIP) ---
        tone_labels = [
            "very light pale skin tone",
            "light beige skin tone",
            "medium tan skin tone",
            "olive brown skin tone",
            "dark brown skin tone",
            "very dark brown skin tone",
        ]
        hair_labels = [
            "skin with visible thick body hair",
            "bare skin without visible thick hair",
        ]
        age_labels = [
            "older skin with deep aging wrinkles",
            "youthful skin without aging wrinkles",
        ]

        # NEW: Specifically check for minor cosmetic conditions
        minor_labels = [
            "clear smooth skin without blemishes",
            "skin with common acne, pimples, or redness",
            "skin with visible pores, texture, or scars",
            "skin with minor dark spots, freckles, or hyperpigmentation",
        ]

        tone_res = classifier(image, candidate_labels=tone_labels)
        hair_res = classifier(image, candidate_labels=hair_labels)
        age_res = classifier(image, candidate_labels=age_labels)
        minor_res = classifier(image, candidate_labels=minor_labels)

        profile_data = {
            "skin_tone": tone_res[0]["label"],
            "hair_presence": hair_res[0]["label"],
            "texture_and_age": age_res[0]["label"],
            "minor_condition": minor_res[0]["label"],  # Added to JSON
            "clip_gatekeeper_confidence": round(val_results[0]["score"], 3),
            "confidence_scores": {
                "skin_tone": round(tone_res[0]["score"], 3),
                "hair_presence": round(hair_res[0]["score"], 3),
                "texture_and_age": round(age_res[0]["score"], 3),
                "minor_condition": round(minor_res[0]["score"], 3),
            },
        }

        # --- STAGE 3: THE MEDICAL SPECIALIST (PyTorch) ---
        # PyTorch now ALWAYS runs on every valid skin image to determine health status.
        if disease_model is not None:
            # 1. Preprocess the image for PyTorch
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            # 2. Make Prediction
            with torch.no_grad():
                outputs = disease_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

            top_class = DISEASE_CLASSES[predicted_idx.item()]
            top_confidence = round(float(confidence.item()), 3)

            # 3. Add debug info (Your mobile app ignores this, but Postman shows it)
            all_probs = {
                DISEASE_CLASSES[i]: round(float(probabilities[i].item()), 3)
                for i in range(4)
            }
            profile_data["model_2_debug"] = all_probs

            # 4. Set the final Health Status based strictly on your PyTorch model
            if top_class == "normal":
                profile_data["health_status"] = (
                    "normal healthy skin (verified by medical model)"
                )
            else:
                profile_data["health_status"] = (
                    "skin with abnormal lesions, moles, or dark spots"
                )
                profile_data["disease_prediction"] = {
                    "detected_class": top_class,
                    "confidence_score": top_confidence,
                }

        # --- STAGE 4: THE DEEPSEEK LLM ---
        if stream:
            return StreamingResponse(
                generate_llm_stream(is_valid=True, profile_data=profile_data),
                media_type="text/event-stream",
                headers=stream_headers,
            )
        else:
            return await generate_llm_json(is_valid=True, profile_data=profile_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------
# LLM GENERATORS
# ---------------------------------------------------------


def get_prompts(is_valid: bool, profile_data: dict):
    """Dynamically builds the prompt based on whether a disease was detected"""

    if not is_valid:
        system_prompt = """
        You are Dr. Domico. 
        The system rejected the user's uploaded image because it either does not contain human skin, is an inanimate object/illustration, or is unreadable.
        Keep your response under 2 sentences. Politely inform the user that the image might not contain a recognizable view of human skin, and ask them to upload a clear, focused photograph of the affected area.
        """
        user_prompt = "Generate the rejection response."
        return system_prompt, user_prompt

    # Check if the PyTorch model flagged a disease
    has_disease = "disease_prediction" in profile_data

    if has_disease:
        # 🚨 THE SERIOUS MEDICAL PROMPT
        system_prompt = """
        You are Dr. Domico, an empathetic, professional dermatologist.
        You are reviewing a patient's skin image alongside your AI screening tool.
        
        CRITICAL RULES:
        1. PRIMARY FOCUS: The AI detected a potential medical condition (see 'disease_prediction'). You MUST prioritize discussing this finding, its confidence level, and strongly urge an in-person clinical biopsy/evaluation.
        2. SECONDARY FOCUS: Briefly acknowledge their 'minor_condition' (like acne or dark spots) and 'skin_tone' in 1 short sentence so the patient knows you closely analyzed their specific skin, but do NOT let it distract from the main medical warning.
        3. MEDICAL DISCLAIMER: Always end with a professional medical disclaimer reminding them this is a screening tool.
        
        Speak directly to the patient in short, readable paragraphs using Markdown formatting.
        """
    else:
        # 🧴 THE COSMETIC / HEALTHY SKIN PROMPT
        system_prompt = """
        You are Dr. Domico, an empathetic, professional dermatologist.
        You are reviewing a patient's skin image alongside your AI screening tool.
        
        CRITICAL RULES:
        1. PRIMARY FOCUS: The medical AI verified this as healthy skin with no severe lesions. Deliver this reassuring news first.
        2. DETAILED COSMETIC ANALYSIS: Dive into the specific details provided by the AI (look at 'minor_condition', 'skin_tone', and 'texture_and_age'). Give personalized, friendly skincare advice based EXACTLY on these minor conditions (e.g., how to handle their specific acne, pores, or how great their clear skin looks).
        3. TONE: Warm, helpful, and highly personalized. Make the user feel like you truly analyzed every detail of their unique skin characteristics.
        4. MEDICAL DISCLAIMER: Always end with a brief disclaimer that this is an AI tool and not a replacement for an in-person doctor visit.
        
        Speak directly to the patient in short, readable paragraphs using Markdown formatting.
        """

    user_prompt = f"Here is the clinical data extracted from the image: {profile_data}. Please provide your consultation."

    return system_prompt, user_prompt


async def generate_llm_stream(is_valid: bool, profile_data: dict):
    """Yields chunked text for the live typing effect (Mobile App)"""
    initial_data = {"type": "metadata", "is_skin": is_valid, "profile": profile_data}
    yield f"data: {json.dumps(initial_data)}\n\n"

    system_prompt, user_prompt = get_prompts(is_valid, profile_data)

    try:
        response = await llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=350,
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


async def generate_llm_json(is_valid: bool, profile_data: dict):
    """Returns a single JSON block instantly (For Postman Testing)"""
    system_prompt, user_prompt = get_prompts(is_valid, profile_data)

    try:
        response = await llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=350,
            temperature=0.7,
            stream=False,
        )

        final_text = response.choices[0].message.content

        return JSONResponse(
            content={
                "is_skin": is_valid,
                "profile": profile_data,
                "ai_response": final_text,
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
