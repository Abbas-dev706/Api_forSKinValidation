# Use the official Python 3.12 slim image to match your local environment
FROM python:3.12-slim

# Set up a new user named "user" with user ID 1000
# (Hugging Face Spaces strictly require running as a non-root user)
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory and update PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container
WORKDIR $HOME/app

# Copy the requirements file first to leverage Docker build cache
# Ensure the files are owned by the non-root user
COPY --chown=user requirements.txt $HOME/app/

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (main.py, .pth files, etc.)
COPY --chown=user . $HOME/app/

# Expose the specific port Hugging Face routes traffic to
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn.
# This assumes your FastAPI instance in main.py is named 'app'.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]