# Technyder RAG System

This project implements a **Retrieval-Augmented Generation (RAG) system** for the [technyder.co](https://technyder.co) website.  
It uses **PostgreSQL with pgvector** to store scraped web pages, generate embeddings (via Google/OpenAI), and answer natural language questions with relevant context.

---

## 🚀 Project Overview

1. **Scraping** – Web pages are crawled from technyder.co and stored in the database (`scraped_pages` table).  
2. **Embeddings** – Each page’s title, content, and combined representation are converted into vector embeddings.  
3. **Storage** – Vectors are stored in PostgreSQL using the [`pgvector`](https://github.com/pgvector/pgvector) extension.  
4. **Querying** – User questions are embedded and compared against stored embeddings using cosine similarity.  
5. **RAG** – The most relevant pages are retrieved and passed to an LLM (Google Gemini / OpenAI GPT) for answer generation.

---

## 🛠️ Database Setup

1. Install **PostgreSQL** with the `pgvector` extension: using the dockers yaml file provided 
   ```bash
   CREATE EXTENSION IF NOT EXISTS vector;



2. CREATE TABLE scraped_pages (
    id SERIAL PRIMARY KEY,
    url VARCHAR(2048) UNIQUE NOT NULL,
    title TEXT,
    content TEXT,
    meta_description TEXT,
    h1_tags TEXT[],
    h2_tags TEXT[],
    content_type VARCHAR(100),
    word_count INTEGER,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    page_hash VARCHAR(64),
    metadata JSONB,
    
    -- Embedding columns
    title_embedding VECTOR(768),
    content_embedding VECTOR(768),
    combined_embedding VECTOR(768),
    embeddings_generated BOOLEAN DEFAULT FALSE,
    embedding_model VARCHAR(100) DEFAULT 'google-text-embedding-004',
    embedding_created_at TIMESTAMP
);



3. pip install -r requirements.txt



4. Run "streamlit run streamlit_app.py" to start streamlit app.
















