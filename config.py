import os
from pathlib import Path
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


PERSIST_DIR = "./chroma_store"
UPLOAD_DIR = "./uploads"
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Models
EMBEDDINGS_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-pro"
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDINGS_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY")
)
# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Flags for language UI
LANG_FLAGS = {
    "English": "🇬🇧",
    "German": "🇩🇪",
    "French": "🇫🇷",
    "Italian": "🇮🇹",
    "Portuguese": "🇵🇹",
    "Hindi": "🇮🇳",
    "Spanish": "🇪🇸",
    "Thai": "🇹🇭"
}

# Gemini LLM call
def call_gemini(prompt):
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text



