import time
import streamlit as st
from config import call_gemini

def ask_gemini(context, question, preferred_lang):
    prompt = f"""
You are a multilingual assistant.
Read the provided context, which may include multiple languages and scripts.
Answer strictly in {preferred_lang}.

When answering:
- Translate ideas faithfully rather than word-for-word
- Use natural, idiomatic expressions in {preferred_lang}
- Adapt dates, numbers, units, and currency to {preferred_lang} locale conventions
- Preserve domain terminology and proper nouns from the context
- If a concept is culture-specific, briefly explain it for {preferred_lang} readers
- If the information is not present in the context, say: "Not found in the provided documents."

Context:
{context}

Question:
{question}
"""
    return call_gemini(prompt)

def typewriter_effect(text, delay=0.02):
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(delay)
