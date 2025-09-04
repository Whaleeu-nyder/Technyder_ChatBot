import streamlit as st
from rag_service import GeminiRAG  # <- put your RAG class code in rag_service.py
import os
from dotenv import load_dotenv
# === Load environment variables ===
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "mydb"),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
    "port": int(os.getenv("DB_PORT", 5432))
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TABLE_NAME = os.getenv("TABLE_NAME", "scraped_pages")
# Streamlit GUI function
def run_rag_app():
    st.set_page_config(page_title="Technyder RAG Search", layout="wide")

    st.title("🔎 Technyder RAG with Gemini + pgvector")
    st.markdown("Ask questions based on content scraped from **technyder.co**")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Database config
    st.sidebar.subheader("Database Settings")
    db_config = DB_CONFIG
    # API Key
    gemini_api_key = GEMINI_API_KEY
    # Search settings
    st.sidebar.subheader("Search Settings")
    table_name = st.sidebar.text_input("Table name", f"{TABLE_NAME}")
    embedding_field = st.sidebar.selectbox(
        "Embedding Field",
        ["combined_embedding", "title_embedding", "content_embedding"]
    )
    max_sources = st.sidebar.slider("Max Sources", 1, 10, 5)

    # Question input
    st.subheader("Ask a Question About Technyder")
    question = st.text_area(
    "Enter your question:",
    st.session_state.get("")
)
    if st.button("Ask"):
        if not gemini_api_key:
            st.error("Please provide your Gemini API Key in the sidebar.")
        else:
            # Init RAG
            rag = GeminiRAG(db_config, gemini_api_key, table_name)

            with st.spinner("Searching and generating answer..."):
                result = rag.ask_question(
                    question,
                    max_sources=max_sources,
                    embedding_field=embedding_field
                )

            # Show Answer
            st.subheader("Answer")
            st.write(result["answer"])
            st.caption(f"Confidence: {result['confidence']:.2f}")

            # Show Sources
            st.subheader("Sources")
            if result["sources"]:
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**[{i}. {src['title']}]({src['url']})** — relevance: {src['relevance_score']:.2f}")
            else:
                st.info("No sources found.")

# Run app
if __name__ == "__main__":
    run_rag_app()
