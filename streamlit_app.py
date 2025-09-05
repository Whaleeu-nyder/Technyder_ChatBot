import streamlit as st
from rag_service import GeminiRAG  # <- put your RAG class code in rag_service.py
import os
from dotenv import load_dotenv
# === Load environment variables ===
load_dotenv()

import streamlit as st

DB_CONFIG = {
    "host": st.secrets["database"]["host"],
    "database": st.secrets["database"]["name"],
    "user": st.secrets["database"]["user"],
    "password": st.secrets["database"]["password"],
    "port": int(st.secrets["database"]["port"]),
}

GEMINI_API_KEY = st.secrets["api"]["gemini_key"]
TABLE_NAME = st.secrets["table"]["name"]

# Streamlit GUI function
def run_rag_app():
    st.set_page_config(page_title="Technyder RAG Search", layout="wide")

    # --- Brand styling (navy/peach) ---
    primary_navy = "#0f2a44"  # deep navy
    accent_peach = "#e2a26f"  # peach/copper
    light_bg = "#f7f8fb"

    st.markdown(
        """
        <style>
        /* App background */
        .main > div {background: var(--light-bg);} 
        :root { --navy: %(navy)s; --peach: %(peach)s; --light-bg: %(light_bg)s; }

        /* Header band */
        .tech-header { 
          background: linear-gradient(90deg, var(--navy) 0%%, var(--navy) 65%%, var(--peach) 65%%, var(--peach) 100%%);
          padding: 18px 24px; border-radius: 10px; color: #fff; margin-bottom: 16px;
        }
        .tech-header .brand {
          font-weight: 900; letter-spacing: 1px; font-size: 22px; display: inline-flex; align-items: center; gap: 10px;
        }
        .tech-header .brand .t-badge { 
          background: #fff; color: var(--navy); font-weight: 900; padding: 4px 8px; border-radius: 6px;
        }
        .tech-sub { color: rgba(255,255,255,0.85); margin-left: 8px; font-weight: 500; }

        /* Cards */
        .tech-card { background: #ffffff; border: 1px solid rgba(15,42,68,0.08); border-radius: 12px; padding: 20px; }
        .tech-accent { border-top: 4px solid var(--peach); }
        .tech-cta button[kind="secondary"] { border-color: var(--peach); }

        /* Inputs */
        textarea, input { border-radius: 10px !important; }

        /* Footer */
        .tech-footer { text-align:center; color:#4c5b6a; margin-top: 10px; }
        .tech-footer .bar { height: 6px; background: linear-gradient(90deg, var(--peach) 0%%, var(--navy) 60%%); border-radius: 6px; margin: 28px auto 8px; max-width: 420px; }
        </style>
        """ % {"navy": primary_navy, "peach": accent_peach, "light_bg": light_bg},
        unsafe_allow_html=True,
    )

    # Header inspired by the mockup
    st.markdown(
        """
        <div class="tech-header">
          <div class="brand">
            <span class="t-badge">T</span>
            <span>TECHNYDER</span>
            <span class="tech-sub">RAG Search</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for configuration
    

    # Database config
    db_config = DB_CONFIG
    # API Key
    gimini_note = "API key is loaded from secrets"
    gemini_api_key = GEMINI_API_KEY
    # Search settings
    st.sidebar.subheader("Search Settings")
    table_name =  f"{TABLE_NAME}"
    embedding_field = "combined_embedding"
    
    #t.sidebar.selectbox("Embedding Field",["combined_embedding", "title_embedding", "content_embedding"] )
    max_sources = st.sidebar.slider("Max Sources", 1, 10, 5)

    # Main two-column layout
    left_col, right_col = st.columns([7, 5])

    with left_col:
        st.markdown("<div class='tech-card tech-accent'>", unsafe_allow_html=True)
        st.subheader("Ask a Question About Technyder")
        question = st.text_area(
            "Enter your question:",
            value=st.session_state.get("question_value", ""),
            key="question_value",
            height=140,
        )
        ask = st.button("Ask", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if ask:
            if not gemini_api_key:
                st.error("Please provide your Gemini API Key in secrets.")
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
                st.markdown("<div class='tech-card'>", unsafe_allow_html=True)
                st.subheader("Answer")
                st.write(result["answer"])
                st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='tech-card'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer with brand bar
    st.markdown(
        """
        <div class="tech-footer">
          <div class="bar"></div>
          <div>© Technyder • Built with RAG and Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Run app
if __name__ == "__main__":
    run_rag_app()