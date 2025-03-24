# News Summarization & Hindi Text-to-Speech App 🚀

This is a simple yet powerful tool that extracts news about a company, analyzes its sentiment, compares multiple articles, and even reads out the key takeaways in Hindi! 📢📰

## 🎯 Features

- **News Extraction** 📰 - Fetches the latest news about a company
- **Sentiment Analysis** 😊😐😡 - Understands if the news is positive, neutral, or negative
- **Comparative Analysis** 📊 - Compares sentiment trends across different articles
- **Text-to-Speech in Hindi** 🎙️🇮🇳 - Converts summaries into Hindi audio
- **Easy-to-Use UI** 💻 - Simple interface using Streamlit

---

## 🛠️ Tech Stack

- **Backend**: Flask API
- **Frontend**: Streamlit
- **ML/NLP**: Hugging Face Transformers, LangChain
- **Translation**: Deep Translator
- **Text-to-Speech**: gTTS (Google Text-to-Speech)

---

## 🔧 Setup & Installation

### 1️⃣ Prerequisites

- **Python 3.9+** installed
- A **Hugging Face API token** (for LLM access) 🔑

### 2️⃣ Clone This Repo

```bash
# Clone the repository
git clone https://github.com/roybishal362/news_summarization_app.git
cd news_summarization_app
```

### 3️⃣ Set Up a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Set Up Your API Key (IMPORTANT ⚠️)

This step is **super important!** You need to add your Hugging Face API token so the app can access LLM features.

```bash
cp .env.example .env
```
Then, **edit the `.env` file** and add your Hugging Face API token like this:

```
HUGGINGFACE_API_TOKEN=your_secret_token_here
```

### 6️⃣ Run the App 🏃‍♂️

```bash
bash run.sh
```

or manually start the API and frontend separately:

```bash
# Start the backend
python api.py
```

```bash
# Start the frontend (Streamlit)
streamlit run app.py
```

Now, open your browser and visit: **`http://localhost:7860`** 🎉

---

## 🐳 Running with Docker (Optional)

Prefer containers? You can run this with Docker too! 🐳

```bash
# Build the Docker image
docker build -t news-analyzer .

# Run the container
docker run -p 7860:7860 -e HUGGINGFACE_API_TOKEN=your_token_here news-analyzer
```

Then, open **`http://localhost:7860`** and start exploring!

---

## 🎮 How to Use

1. **Enter a company name** or select one from the dropdown.
2. Click **"Analyze"** and let the magic happen! ✨
3. Explore the results:
   - 📰 **Article Summaries** - Quick insights from fetched news
   - 📊 **Sentiment Analysis** - See if the news is positive, negative, or neutral
   - 🔍 **Comparative Analysis** - Compare sentiment trends across sources
   - 🎙️ **Hindi TTS** - Listen to the summary in Hindi

---

## 📡 API Endpoints

The backend provides a RESTful API with the following endpoints:

### 🔍 Health Check

- **Endpoint**: `/api/health`
- **Method**: `GET`
- **Response**: `{ "status": "healthy" }`

### 📰 News Analysis

- **Endpoint**: `/api/news-analysis`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "company_name": "Tesla"
  }
  ```
- **Response**: Returns summarized news with sentiment analysis

---

## 🚧 Limitations & Notes

1. 🕷 **Web scraping** - If news websites change, scraping might break.
2. 🧠 **LLM responses** - Generated content may vary depending on the model.
3. 🌍 **Translation accuracy** - Hindi translation depends on Deep Translator.
4. 🔊 **TTS quality** - Audio is based on gTTS, so pronunciation might not be perfect.

---

## 🚀 Deployment

This app is deployed on Hugging Face Spaces! Check it out here: **https://huggingface.co/spaces/roybishal943/NewsSummarization**

---

## ❤️ Contributing

Found a bug? Want to add a feature? Feel free to open an issue or submit a pull request! 🚀

