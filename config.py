import os
from pathlib import Path
import google.generativeai as genai

PERSIST_DIR = "./chroma_store"
UPLOAD_DIR = "./uploads"
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Models
EMBEDDINGS_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-pro"

# Configure Gemini
genai.configure(api_key="AIzaSyBLn0HDt3j4NSag9H9wGlcxlAkgkrILJ_g")

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
