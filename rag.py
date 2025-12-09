import cohere
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

load_dotenv() 

co = cohere.Client(os.getenv("COHERE_API_KEY"))


# --------- PDF LOADING ----------
def load_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# --------- CHUNKING ----------
def chunk_text(text, chunk_size=350):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# --------- EMBEDDINGS ----------
def embed_chunks(chunks):
    response = co.embed(
        texts=chunks,
        model="embed-multilingual-v3.0"
    )
    return np.array(response.embeddings)


# --------- SEMANTIC RETRIEVAL ----------
def retrieve(query, chunk_embeddings, chunks, top_k=3):
    q_emb = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0"
    ).embeddings[0]

    scores = np.dot(chunk_embeddings, q_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
    )

    top_idx = np.argsort(scores)[-top_k:][::-1]
    retrieved = [(chunks[i], float(scores[i])) for i in top_idx]
    return retrieved


# --------- LLM ANSWERING ----------
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join([chunk for chunk, score in retrieved_chunks])

    prompt = f"""
You are a helpful assistant. Answer the question ONLY using the context below.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}
Answer:
"""

    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300,
        temperature=0
    )
    return response.generations[0].text.strip()
