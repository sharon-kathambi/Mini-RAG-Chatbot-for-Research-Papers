import streamlit as st
from rag import load_pdf, chunk_text, embed_chunks, retrieve, generate_answer

st.set_page_config(page_title="Mini RAG Cohere Chatbot", page_icon="🤖")
st.title("🤖 Mini RAG Chatbot — Powered by Cohere")
st.write("Upload a PDF and ask questions grounded in the document.")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    st.success("PDF uploaded successfully. Processing...")
    
    text = load_pdf(uploaded_pdf)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    st.success("Document indexed! You can now ask questions.")

    question = st.text_input("Ask a question about the document:")

    if question:
        st.write("🔍 Retrieving relevant context...")
        retrieved = retrieve(question, embeddings, chunks)

        st.write("🧠 Generating answer...")
        answer = generate_answer(question, retrieved)

        st.subheader("📘 Answer:")
        st.write(answer)

        with st.expander("View Retrieved Chunks"):
            for chunk, score in retrieved:
                st.write(f"**Score:** {score:.4f}")
                st.write(chunk)
                st.write("---")
