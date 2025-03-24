FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install langchain langchain_community huggingface_hub

# Copy project files
COPY . .

# Set environment variables
ENV PORT=7860
ENV HOST=0.0.0.0
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

# Expose port for the app
EXPOSE 7860

# Start the application
CMD ["bash", "run.sh"]