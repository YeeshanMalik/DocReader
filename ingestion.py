import os
import uuid
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import PERSIST_DIR, UPLOAD_DIR, embeddings

def save_upload(file):
    suffix = Path(file.name).suffix.lower()
    fname = f"{uuid.uuid4().hex}{suffix}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(file.read())
    return fpath

def load_document(path):
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    if path.lower().endswith(".docx"):
        return Docx2txtLoader(path).load()
    if path.lower().endswith(".txt"):
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError("Unsupported file type")

def split_docs(docs, chunk_size=1200, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def get_vectordb():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def add_docs_to_vectordb(chunks, source_name):
    metad = []
    for d in chunks:
        m = d.metadata or {}
        m["source"] = source_name
        metad.append(m)
    vectordb = get_vectordb()
    vectordb.add_texts(texts=[c.page_content for c in chunks], metadatas=metad)
    vectordb.persist()
