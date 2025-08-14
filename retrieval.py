from ingestion import get_vectordb

def retrieve(query, k=6):
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)
from ingestion import get_vectordb

def retrieve_with_scores(query, k=6):
    vectordb = get_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    results = vectordb.similarity_search_with_score(query, k=k)
    return results  # returns [(Document, score), ...]


def build_context(docs, max_chars=3500):
    out = []
    total = 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        seg = f"[{src}]\n{d.page_content}\n"
        if total + len(seg) > max_chars:
            break
        out.append(seg)
        total += len(seg)
    return "\n".join(out)
