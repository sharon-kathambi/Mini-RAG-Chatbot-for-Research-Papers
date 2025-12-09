# Mini-RAG-Chatbot-for-Research-Papers
A lightweight **Retrieval-Augmented Generation (RAG)** system that allows you to **upload any research paper (PDF)** and ask questions about it — similar to interacting with a notebook-style LLM.
The chatbot answers using **only the content in the paper**, making it ideal for grounded academic and technical exploration.

This project uses **Cohere Embed** for semantic retrieval and **Cohere Command** for context-aware generation, demonstrating how to build a practical end-to-end RAG pipeline with just a few lines of code.

---

## 🌟 Why I Built This

As a machine learning researcher working on my master’s thesis and reading dozens of dense academic papers, I wanted a simple tool that could **help me query a research paper directly**, without relying on generic or hallucinated answers.

I built this project to:

* Quickly **understand complex papers** by asking targeted questions
* Use LLMs as a “research companion,” grounded in the text
* Demonstrate practical skills in **embeddings, semantic retrieval, and RAG**
* Explore Cohere’s capabilities in building **real-world, production-ready NLP tools**
* Create an intuitive, fast, and lightweight alternative to notebook-style LLM assistants

This project also showcases the kind of applied ML/NLP engineering I enjoy — building tools that combine strong research with practical impact.


## 🚀 Features

* 📄 **Upload any research paper (PDF)**
* ✂️ Automatic text extraction + chunking
* 🔍 **Semantic search** using Cohere’s embedding model
* 🧠 **RAG pipeline** to retrieve the most relevant chunks
* ✨ **Grounded answers** generated via Cohere Command
* 📚 Evidence view: see the chunks used to generate the answer
* 🖥️ Clean, simple Streamlit interface


## 🧠 Tech Stack

* **Cohere Embed** — semantic embeddings
* **Cohere Command R+** — grounded text generation
* **Streamlit** — interactive UI
* **PyPDF2** — PDF text extraction
* **NumPy** — similarity calculations


## 🛠️ How It Works

1. **PDF Upload**
   The user uploads a research paper.

2. **Text Extraction & Chunking**
   The document is split into digestible text chunks (~350 tokens each).

3. **Embedding Generation**
   Each chunk is embedded using Cohere’s multilingual embedding model.

4. **Semantic Retrieval**
   A question is embedded and matched via cosine similarity to the closest chunks.

5. **Grounded Generation**
   The retrieved chunks + question are fed into Cohere Command R+ to produce a context-aware answer.

6. **Answer Display**
   The model’s answer and the supporting chunks are shown in the UI.



## 📂 Project Structure

```
mini-rag-cohere-chatbot/
│
├── app.py            # Streamlit UI
├── rag.py            # Core RAG logic
├── requirements.txt
└── README.md
```



## ▶️ Running the App

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run the Streamlit app**

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📝 Example Use Cases

* Ask a research paper:

  * “What problem does this paper aim to solve?”
  * “Summarize the method section.”
  * “What datasets or benchmarks were used?”
  * “Explain the core intuition in simpler terms.”

* Build your own personal academic assistant

* Use it to review documentation, handbooks, or technical guides

* Enhance productivity as a student, researcher, or engineer



## 🌍 Future Improvements

* Support multiple documents
* Improved chunking strategies (overlap, sentence-level)
* Reranking with Cohere’s **Rerank** model
* PDF highlighting of retrieved evidence
* Notebook/Colab version

