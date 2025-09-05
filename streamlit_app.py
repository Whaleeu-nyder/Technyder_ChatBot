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
\tst.set_page_config(page_title="Technyder RAG Search", layout="wide")

\t# --- Brand styling (navy/peach) ---
\tprimary_navy = "#0f2a44"  # deep navy
\taccent_peach = "#e2a26f"  # peach/copper
\tlight_bg = "#f7f8fb"

\tst.markdown(
\t\t"""
\t\t<style>
\t\t/* App background */
\t\t.main > div {background: var(--light-bg);} 
\t\t:root { --navy: %(navy)s; --peach: %(peach)s; --light-bg: %(light_bg)s; }

\t\t/* Header band */
\t\t.tech-header { 
\t\t  background: linear-gradient(90deg, var(--navy) 0%%, var(--navy) 65%%, var(--peach) 65%%, var(--peach) 100%%);
\t\t  padding: 18px 24px; border-radius: 10px; color: #fff; margin-bottom: 16px;
\t\t}
\t\t.tech-header .brand {
\t\t  font-weight: 900; letter-spacing: 1px; font-size: 22px; display: inline-flex; align-items: center; gap: 10px;
\t\t}
\t\t.tech-header .brand .t-badge { 
\t\t  background: #fff; color: var(--navy); font-weight: 900; padding: 4px 8px; border-radius: 6px;
\t\t}
\t\t.tech-sub { color: rgba(255,255,255,0.85); margin-left: 8px; font-weight: 500; }

\t\t/* Cards */
\t\t.tech-card { background: #ffffff; border: 1px solid rgba(15,42,68,0.08); border-radius: 12px; padding: 20px; }
\t\t.tech-accent { border-top: 4px solid var(--peach); }
\t\t.tech-cta button[kind="secondary"] { border-color: var(--peach); }

\t\t/* Inputs */
\t\ttextarea, input { border-radius: 10px !important; }

\t\t/* Footer */
\t\t.tech-footer { text-align:center; color:#4c5b6a; margin-top: 10px; }
\t\t.tech-footer .bar { height: 6px; background: linear-gradient(90deg, var(--peach) 0%%, var(--navy) 60%%); border-radius: 6px; margin: 28px auto 8px; max-width: 420px; }
\t\t</style>
\t\t""" % {"navy": primary_navy, "peach": accent_peach, "light_bg": light_bg},
\t\tunsafe_allow_html=True,
\t)

\t# Header inspired by the mockup
\tst.markdown(
\t\t"""
\t\t<div class="tech-header">
\t\t  <div class="brand">
\t\t    <span class="t-badge">T</span>
\t\t    <span>TECHNYDER</span>
\t\t    <span class="tech-sub">RAG Search</span>
\t\t  </div>
\t\t</div>
\t\t""",
\t\tunsafe_allow_html=True,
\t)

\t# Sidebar for configuration
\tst.sidebar.header("Configuration")

    # Database config
\tst.sidebar.subheader("Database Settings")
\tdb_config = DB_CONFIG
    # API Key
\tgimini_note = "API key is loaded from secrets"
\tgemini_api_key = GEMINI_API_KEY
    # Search settings
\tst.sidebar.subheader("Search Settings")
\ttable_name = st.sidebar.text_input("Table name", f"{TABLE_NAME}")
\tembedding_field = st.sidebar.selectbox(
\t\t"Embedding Field",
\t\t["combined_embedding", "title_embedding", "content_embedding"]
\t)
\tmax_sources = st.sidebar.slider("Max Sources", 1, 10, 5)

\t# Main two-column layout
\tleft_col, right_col = st.columns([7, 5])

\twith left_col:
\t\tst.markdown("<div class='tech-card tech-accent'>", unsafe_allow_html=True)
\t\tst.subheader("Ask a Question About Technyder")
\t\tquestion = st.text_area(
\t\t\t"Enter your question:",
\t\t\tvalue=st.session_state.get("question_value", ""),
\t\t\tkey="question_value",
\t\t\theight=140,
\t\t)
\t\task = st.button("Ask", type="primary")
\t\tst.markdown("</div>", unsafe_allow_html=True)

\t\tif ask:
\t\t\tif not gemini_api_key:
\t\t\t\tst.error("Please provide your Gemini API Key in secrets.")
\t\t\telse:
\t\t\t\t# Init RAG
\t\t\t\trag = GeminiRAG(db_config, gemini_api_key, table_name)

\t\t\t\twith st.spinner("Searching and generating answer..."):
\t\t\t\t\tresult = rag.ask_question(
\t\t\t\t\t\tquestion,
\t\t\t\t\t\tmax_sources=max_sources,
\t\t\t\t\t\tembedding_field=embedding_field
\t\t\t\t\t)

\t\t\t\t# Show Answer
\t\t\t\tst.markdown("<div class='tech-card'>", unsafe_allow_html=True)
\t\t\t\tst.subheader("Answer")
\t\t\t\tst.write(result["answer"])
\t\t\t\tst.markdown("</div>", unsafe_allow_html=True)

\twith right_col:
\t\tst.markdown("<div class='tech-card'>", unsafe_allow_html=True)
\t\tst.subheader("Search Options")
\t\tst.caption("%s" % gimini_note)
\t\tst.write(":orange[Embedding field]", embedding_field)
\t\tst.write(":orange[Max sources]", max_sources)
\t\tst.write(":orange[Table]", table_name)
\t\tst.markdown("</div>", unsafe_allow_html=True)

\t# Footer with brand bar
\tst.markdown(
\t\t"""
\t\t<div class="tech-footer">
\t\t  <div class="bar"></div>
\t\t  <div>© Technyder • Built with RAG and Streamlit</div>
\t\t</div>
\t\t""",
\t\tunsafe_allow_html=True,
\t)
          

# Run app
if __name__ == "__main__":
    run_rag_app()
