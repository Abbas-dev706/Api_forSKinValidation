# Use the official Python 3.12 slim image to match your local environment
FROM python:3.12-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory and update PATH
# ADDED: HF_HOME environment variable to route transformer model downloads to a writable folder
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/app/.cache/huggingface

# Set the working directory inside the container
WORKDIR $HOME/app

# ADDED: Explicitly create the cache directory so the CLIP model has a safe place to download
RUN mkdir -p $HF_HOME

# Copy the requirements file first to leverage Docker build cache
COPY --chown=user requirements.txt $HOME/app/

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (main.py, .pth files, etc.)
COPY --chown=user . $HOME/app/

# Expose the specific port Hugging Face routes traffic to
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]