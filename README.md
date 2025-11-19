# PDF Question Answering System Using Transformer Models and Semantic Retrieval

A lightweight, transformer-based **PDF Question Answering (QA) system** that extracts relevant answers directly from PDF documents using **Sentence-BERT**, **FAISS**, and a **Streamlit** interface.

This system allows users to upload a PDF and ask natural-language questions. The model performs semantic retrieval and generates accurate answers grounded in the document content.

---

## ⭐ Features

- 📄 **PDF Upload & Text Extraction** using `pdfplumber` / `PyPDF2`
- 🔍 **Semantic Search** with FAISS vector indexing
- 🧠 **Transformer-Based QA Models** (DistilBERT / T5-small)
- 🔤 **Sentence Embeddings** via Sentence-BERT
- 🪜 **Chunking & Metadata Tracking**
- ⚡ **Fast Answer Retrieval**
- 💻 **Streamlit Web Interface**

---

## 🏗️ System Architecture

PDF Upload → Text Extraction → Preprocessing → Chunking
→ Sentence-BERT Embeddings → FAISS Vector Search
→ Transformer QA Model → Answer + Source Chunk
→ Streamlit UI


---

## 🧪 Tech Stack

| Component | Library/Model |
|----------|----------------|
| UI | Streamlit |
| Embeddings | Sentence-BERT (`all-MiniLM-L6-v2`) |
| QA Model | DistilBERT / T5-small |
| Vector Search | FAISS |
| PDF Extraction | pdfplumber / PyPDF2 |
| NLP Preprocessing | NLTK |
| Language | Python |

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/PDF-Question-Answering-System.git
cd PDF-Question-Answering-System
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```bash
streamlit run project_code.py
```

## 🧠 How It Works
### 1. PDF Extraction

  - Extracts raw text using pdfplumber
  
  - Falls back to PyPDF2 for simpler files
  
  - Cleans text using normalization and regex

### 2. Chunking

  - Splits text into overlapping segments (512 tokens)
  
  - Maintains page references for transparency

### 3. Semantic Embeddings

  - Uses Sentence-BERT to encode chunks into vectors
  
  - Produces 384-dimensional embeddings

### 4. FAISS Vector Search

  - Stores embeddings in FAISS for fast retrieval
  
  - Similarity search takes <100 ms

### 5. Question Answering

  - Embeds the user’s question
  
  - Retrieves top relevant chunks
  
  - Answer generated using DistilBERT or T5-small

### 6. Streamlit UI

  - Upload PDF
  
  - Ask questions
  
  - View answers + source text
