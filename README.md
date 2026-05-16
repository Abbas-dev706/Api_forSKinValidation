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

```

### 📋 Team Guide: Getting Started with the Dermico API

Welcome to the backend team! We are hosting our FastAPI backend on Hugging Face Spaces. Because we are using Git to push code and dealing with large AI models, you need to do a quick one-time setup before you can pull or push code.

#### 🛑 STEP 1: The One-Time Setup (Do this first)

1. **Create a Hugging Face Account:** Make sure you have signed up at huggingface.co and accepted the invite to our Organization.
2. **Generate an Access Token (Crucial):** Hugging Face does not allow you to push code using your account password.
* Go to **Settings > Access Tokens**.
* Click **Create new token**.
* Select **Write** permissions (or Fine-grained with repo write access).
* **Copy this token and save it somewhere safe.** You will paste this into your terminal when it asks for a password during `git push`.


3. **Install Git LFS:** Because our `skin_disease_final.pth` model is huge, standard Git cannot handle it. Open your terminal and run:

```bash
git lfs install
```


*(If it says command not found, you need to download Git LFS from git-lfs.com first).*

#### ⬇️ STEP 2: Your First Pull (Cloning the Repo)

Once LFS is installed, you can pull the code to your local machine. Open your terminal in the folder where you want the project to live and run:

```bash
# Replace YOUR_ORG_NAME with our actual organization name
git clone https://huggingface.co/spaces/dermico-team/dermico

# Move into the project folder
cd dermico

```

*Note: The download might take a minute because it is pulling the actual `.pth` weights.*

#### 🔄 STEP 3: The Daily Workflow (Pull, Edit, Push)

Before you start writing code for the day, **always pull the latest changes** so you don't overwrite someone else's work:

```bash
git pull origin main
```

When you are done writing your code and want to push it to the live server:

```bash
# 1. Stage your changes
git add .

# 2. Write a clear commit message
git commit -m "Added new route for user history"

# 3. Push to the live server
git push

```

**Authentication Check:**
When you hit `git push`, the terminal will ask for your credentials:

* **Username:** Your Hugging Face username.
* **Password:** Paste your **Access Token** (it will be invisible as you paste it, just press Enter).

Once the push is successful, Hugging Face will automatically rebuild the Docker container and deploy the new code!