#!/usr/bin/env python3
"""
Gemini RAG Query Service with pgvector
Supports title/content/combined embeddings + hybrid search
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
from typing import List, Dict
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import os
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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    id: int
    url: str
    title: str
    content: str
    relevance_score: float
    word_count: int = 0


class GeminiRAG:
    """Gemini-powered RAG with pgvector"""

    def __init__(self, db_config: Dict[str, str], gemini_api_key: str,
                 table_name: str = "scraped_pages", model: str = "gemini-1.5-flash"):

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model)
        self.embed_model = "models/text-embedding-004"  # Gemini embeddings

        # Database setup
        self.db_config = db_config
        self.table_name = table_name

        logger.info(f"✅ GeminiRAG initialized with embeddings and table: {table_name}")

    def connect_db(self):
        return psycopg2.connect(**self.db_config)

    def embed_query(self, text: str) -> List[float]:
        """Get Gemini embedding for a query"""
        try:
            resp = genai.embed_content(model=self.embed_model, content=text)
            return resp["embedding"]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    def search_content(self, query: str, limit: int = 5,
                       embedding_field: str = "combined_embedding",
                       hybrid: bool = False) -> List[SearchResult]:
        """
        Search using pgvector.
        embedding_field: 'title_embedding', 'content_embedding', or 'combined_embedding'
        hybrid: if True, combines full-text rank + embeddings
        """
        if embedding_field not in ["title_embedding", "content_embedding", "combined_embedding"]:
            raise ValueError(f"Invalid embedding_field: {embedding_field}")

        try:
            query_embedding = self.embed_query(query)
            if not query_embedding:
                return []

            with self.connect_db() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:

                    if hybrid:
                        # Hybrid: combine fulltext + embedding
                        search_query =  f"""
                                            SELECT id, url, title, content, word_count,
                                                ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s)) AS fulltext_score,
                                                1 - ({embedding_field} <=> %s::vector) AS embedding_score,
                                                (0.4 * ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s))
                                                + 0.6 * (1 - ({embedding_field} <=> %s::vector))) AS relevance_score
                                            FROM {self.table_name}
                                            WHERE status = 'active'
                                            ORDER BY relevance_score DESC
                                            LIMIT %s
                                        """
                        cur.execute(search_query, (query, query_embedding, query, query_embedding, limit))

                    else:
                        # Pure embedding search
                    
                        search_query = f"""
                            SELECT id, url, title, content, word_count,
                                    COALESCE(1 - ({embedding_field} <=> %s::vector), 0) AS relevance_score
                                FROM {self.table_name}
                                WHERE status = 'active'
                                AND {embedding_field} IS NOT NULL
                                ORDER BY relevance_score DESC
                                LIMIT %s
                        """
                        cur.execute(search_query, (query_embedding, limit))
                        cur.execute(search_query, (query_embedding, limit))

                    results = cur.fetchall()

                    return [
                        SearchResult(
                            id=row["id"],
                            url=row["url"] or "",
                            title=row["title"] or "No Title",
                            content=row["content"] or "",
                            relevance_score=float(row["relevance_score"]),
                            word_count=row.get("word_count", 0)
                        )
                        for row in results
                    ]

        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []

    def ask_question(self, question: str, max_sources: int = 50,
                     embedding_field: str = "combined_embedding",
                     hybrid: bool = False) -> Dict:
        """RAG pipeline with embeddings + optional hybrid search"""
        try:
            search_results = self.search_content(
                question, limit=max_sources,
                embedding_field=embedding_field,
                hybrid=hybrid
            )
                
            if not search_results or search_results[0].relevance_score < 0.3:
                # fallback: let Gemini answer directly
                response = self.model.generate_content(
                    f"Answer the following question based on your knowledge:\n\n{question}",
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=400,
                        temperature=0.5,
                    )
                )
                return {
                    "answer": response.text,
                    "sources": [],
                    "confidence": 0.5
                }






            # Prepare context
            context_parts = []
            for i, result in enumerate(search_results, 1):
                snippet = result.content[:800] + "..." if len(result.content) > 800 else result.content
                context_parts.append(f"""
[Source {i}]
Title: {result.title}
URL: {result.url}
Content: {snippet}
---""")

            context = "\n".join(context_parts)

            prompt = f"""You are a helpful assistant. Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                )
            )

            sources = [
                {
                    "title": result.title,
                    "url": result.url,
                    "relevance_score": result.relevance_score
                }
                for result in search_results
            ]

            return {
                "answer": response.text,
                "sources": sources,
                "confidence": 0.85   ,
                "num_sources": len(search_results)
            }

        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "sources": [],
                "confidence": 0.0
            }


# === Quick Test ===
if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'mydb',
        'user': 'myuser', 
        'password': 'mypassword',
        'port': 5434
    }
  

    GEMINI_API_KEY = "AIzaSyDhrPLQnZBQmXJ7ufV_F-MefwKiXzRdkRA"
    TABLE_NAME = "scraped_pages"

    rag = GeminiRAG(DB_CONFIG, GEMINI_API_KEY, TABLE_NAME)

    # Pure embedding search
    res = rag.ask_question("Based on the information from technyder.co, what kinds of technology and AI services does Technyder provide—such as custom software, DevOps, automation, machine learning, or chatbot development—and how do these services help businesses improve efficiency or transform digitally?", embedding_field="combined_embedding")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
