# Skinalyze API Backend 🧬

An AI-powered dermatology and skincare assistant backend built with **FastAPI**. This system processes user-uploaded images through a multi-stage AI pipeline to validate skin, extract visual profiles, classify potential diseases, and stream natural language analysis directly to the client.

## 🚀 Architecture & Pipeline

This API uses a "Conditional Routing" architecture to optimize speed and accuracy:
1. **The Gatekeeper (OpenAI CLIP):** Validates if the uploaded image is actually human skin. If not, it politely rejects the image and halts processing to save resources.
2. **The Profiler (OpenAI CLIP):** Extracts visual data including Skin Tone (Fitzpatrick-inspired), Hair Presence, and Skin Texture/Age.
3. **The Disease Specialist (Custom TensorFlow CNN):** If texture or blemishes are detected, a custom `.h5` model evaluates the image for three specific classes: *Melanoma, Basal Cell Carcinoma, and Melanocytic Nevi*.
4. **The LLM Synthesizer (DeepSeek AI):** Converts the raw JSON data and confidence scores into a conversational, medically sensitive text analysis, streamed back to the client in real-time.

---

## 🛠️ Prerequisites

* **Python 3.10+** (Tested on Python 3.12)
* **Windows Users:** You MUST install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) to prevent PyTorch `c10.dll` initialization errors.

---

## 💻 Installation & Setup

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/yourusername/Api_forSKinValidation.git](https://github.com/yourusername/Api_forSKinValidation.git)
cd Api_forSKinValidation