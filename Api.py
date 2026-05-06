"""
skin_api.py
-----------
API for skin detection using skin_gate.py logic
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Import your existing logic
from skin_gate import is_skin_image

app = FastAPI(title="Skin Detection API")


@app.get("/")
def home():
    return {"message": "Skin Detection API is running"}


@app.post("/check-skin/")
async def check_skin(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be an image"}
            )

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply skin detection
        passed, ratio = is_skin_image(image)

        # Response
        return {
            "success": True,
            "is_skin_image": passed,
            "skin_ratio": round(ratio, 4),
            "message": (
                "Valid skin image" if passed
                else "Not enough skin detected"
            )
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )