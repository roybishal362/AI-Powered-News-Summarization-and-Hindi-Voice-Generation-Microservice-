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
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PORT=7860
ENV HOST=0.0.0.0

# Set up ollama
RUN mkdir -p /root/.ollama
RUN python -c "from langchain.llms import Ollama; from langchain.chains import LLMChain; from langchain.prompts import PromptTemplate; llm = Ollama(model='llama3')"

# Expose port for the app
EXPOSE 7860

# Start the application
CMD ["./run.sh"]