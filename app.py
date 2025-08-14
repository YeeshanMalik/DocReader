import streamlit as st
from ingestion import save_upload, load_document, split_docs, add_docs_to_vectordb
from retrieval import retrieve_with_scores
from generation import ask_llama, typewriter_effect
from config import LANG_FLAGS

st.set_page_config(page_title="🌍 Multilingual RAG — Dynamic Mode", layout="wide")

ingest_col, qa_col = st.columns([1, 1.5])

with ingest_col:
    st.subheader("📂 Ingest Documents")
    files = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    chunk_size = st.number_input("Chunk size", min_value=400, max_value=2400, value=1200, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=50, max_value=600, value=200, step=50)
    if st.button("🚀 Process & Index") and files:
        for f in files:
            with st.spinner(f"📥 Processing {f.name}..."):
                path = save_upload(f)
                docs = load_document(path)
                chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                add_docs_to_vectordb(chunks, source_name=f.name)
        st.success("✅ Ingestion complete")

with qa_col:
    st.subheader("💬 Ask a Question")
    q = st.text_area("Your question")
    preferred_lang = st.selectbox("Preferred answer language", list(LANG_FLAGS.keys()))
    top_k = st.slider("Top-K passages", 3, 12, 6, 1)

    if st.button("🔍 Retrieve & Answer"):
        if not q:
            st.error("Please enter a question")
        else:
            with st.spinner("🧠 Embedding and searching..."):
                results = retrieve_with_scores(q, k=top_k)

            if not results:
                st.warning("No results found")
            else:
                with st.spinner("📚 Building context..."):
                    context_text = ""
                    for doc, _ in results:
                        context_text += f"[{doc.metadata.get('source','unknown')}]\n{doc.page_content}\n"

                with st.spinner(f"🤖 Generating answer in {preferred_lang} {LANG_FLAGS.get(preferred_lang,'')}..."):
                    answer = ask_llama(context_text, q, preferred_lang)

                st.markdown(f"### ✨ Answer ({LANG_FLAGS.get(preferred_lang,'')} {preferred_lang})")
                typewriter_effect(answer)

                with st.expander("🧾 Reasoning Trail"):
                    for doc, score in results:
                        pct = round((1 - score) * 100, 2)  # convert distance to % match
                        st.markdown(f"**Doc:** {doc.metadata.get('source','unknown')} | **Relevance:** {pct}%")
                        st.write(doc.page_content)
