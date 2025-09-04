import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from urllib.parse import urljoin, urlparse, urldefrag
import hashlib
import time
import logging
from datetime import datetime
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
import os
from dataclasses import dataclass
import google.generativeai as genai
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PageContent:
    url: str
    title: str
    content: str
    meta_description: str
    h1_tags: List[str]
    h2_tags: List[str]
    word_count: int
    content_hash: str
    metadata: Dict

class WebsiteScraper:
    def __init__(self, db_config: Dict[str, str], max_workers: int = 5):
        self.db_config = db_config
        self.max_workers = max_workers
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Website Content Scraper 1.0'
        })
        
    def connect_db(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def start_scrape_session(self) -> int:
        """Start a new scraping session and return session ID"""
        with self.connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO scrape_sessions (started_at) VALUES (CURRENT_TIMESTAMP) RETURNING id"
                )
                return cur.fetchone()[0]
    
    def update_scrape_session(self, session_id: int, pages_scraped: int, 
                            pages_updated: int, pages_failed: int, status: str = 'completed'):
        """Update scraping session statistics"""
        with self.connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE scrape_sessions 
                    SET completed_at = CURRENT_TIMESTAMP,
                        pages_scraped = %s,
                        pages_updated = %s, 
                        pages_failed = %s,
                        status = %s
                    WHERE id = %s
                """, (pages_scraped, pages_updated, pages_failed, status, session_id))
    
    def get_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Extract URLs from sitemap"""
        try:
            response = self.session.get(sitemap_url)
            response.raise_for_status()
            
            # Parse XML sitemap
            root = ET.fromstring(response.content)
            urls = []
            
            # Handle different sitemap formats
            for elem in root.iter():
                if elem.tag.endswith('loc'):
                    urls.append(elem.text)
                    
            return urls
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
            return []
    
    def scrape_page(self, url: str) -> Optional[PageContent]:
        """Scrape content from a single page"""
        try:
            # Clean URL (remove fragments)
            url = urldefrag(url)[0]
            
            if url in self.visited_urls:
                return None
                
            self.visited_urls.add(url)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract content
            title = soup.find('title')
            title = title.get_text().strip() if title else ''
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '').strip() if meta_desc else ''
            
            # Get headings
            h1_tags = [h1.get_text().strip() for h1 in soup.find_all('h1')]
            h2_tags = [h2.get_text().strip() for h2 in soup.find_all('h2')]
            
            # Get main content
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content).strip()
            
            word_count = len(content.split())
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            metadata = {
                'content_length': len(content),
                'response_status': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'last_modified': response.headers.get('last-modified', ''),
                'scraped_timestamp': datetime.now().isoformat()
            }
            
            return PageContent(
                url=url,
                title=title,
                content=content,
                meta_description=meta_description,
                h1_tags=h1_tags,
                h2_tags=h2_tags,
                word_count=word_count,
                content_hash=content_hash,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split content into overlapping chunks"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            if end < len(content):
                # Find the last sentence boundary within the chunk
                last_period = content.rfind('.', start, end)
                last_newline = content.rfind('\n', start, end)
                last_boundary = max(last_period, last_newline)
                
                if last_boundary > start + chunk_size // 2:
                    end = last_boundary + 1
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(content) else end
            
        return chunks
    
    def save_to_database(self, page_content: PageContent) -> bool:
        """Save scraped content to PostgreSQL database"""
        try:
            with self.connect_db() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check if page already exists
                    cur.execute("SELECT id, page_hash FROM scraped_pages WHERE url = %s", (page_content.url,))
                    existing = cur.fetchone()
                    
                    if existing:
                        # Check if content has changed
                        if existing['page_hash'] == page_content.content_hash:
                            logger.info(f"No changes detected for {page_content.url}")
                            return True
                        
                        # Update existing page
                        cur.execute("""
                            UPDATE scraped_pages 
                            SET title = %s, content = %s, meta_description = %s,
                                h1_tags = %s, h2_tags = %s, word_count = %s,
                                page_hash = %s, metadata = %s, last_updated = CURRENT_TIMESTAMP
                            WHERE url = %s
                            RETURNING id
                        """, (
                            page_content.title, page_content.content, page_content.meta_description,
                            page_content.h1_tags, page_content.h2_tags, page_content.word_count,
                            page_content.content_hash, Json(page_content.metadata), page_content.url
                        ))
                        
                        page_id = cur.fetchone()['id']
                        
                        # Delete old chunks
                        cur.execute("DELETE FROM content_chunks WHERE page_id = %s", (page_id,))
                        
                    else:
                        # Insert new page
                        cur.execute("""
                            INSERT INTO scraped_pages 
                            (url, title, content, meta_description, h1_tags, h2_tags, 
                             word_count, page_hash, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (
                            page_content.url, page_content.title, page_content.content,
                            page_content.meta_description, page_content.h1_tags, page_content.h2_tags,
                            page_content.word_count, page_content.content_hash, Json(page_content.metadata)
                        ))
                        
                        page_id = cur.fetchone()['id']
                    
                    # Create content chunks
                    chunks = self.chunk_content(page_content.content)
                    
                    for i, chunk in enumerate(chunks):
                        token_count = len(chunk.split())  # Simple token estimation
                        cur.execute("""
                            INSERT INTO content_chunks (page_id, chunk_text, chunk_order, chunk_tokens)
                            VALUES (%s, %s, %s, %s)
                        """, (page_id, chunk, i, token_count))
                    
                    logger.info(f"Saved {page_content.url} with {len(chunks)} chunks")
                    return True
                    
        except Exception as e:
            logger.error(f"Error saving {page_content.url} to database: {e}")
            return False
    
    def discover_urls(self, base_url: str, max_depth: int = 3) -> Set[str]:
        """Discover URLs by crawling the website"""
        discovered_urls = set()
        to_visit = [(base_url, 0)]
        visited = set()
        
        while to_visit:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
                
            visited.add(url)
            
            try:
                response = self.session.get(url, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                discovered_urls.add(url)
                
                # Find all internal links
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link['href'])
                    parsed_base = urlparse(base_url)
                    parsed_url = urlparse(full_url)
                    
                    # Only follow internal links
                    if (parsed_url.netloc == parsed_base.netloc and 
                        full_url not in visited and 
                        depth < max_depth):
                        to_visit.append((full_url, depth + 1))
                        
            except Exception as e:
                logger.error(f"Error discovering URLs from {url}: {e}")
                
        return discovered_urls
    
    def scrape_website(self, base_url: str, use_sitemap: bool = True, 
                      discover_urls: bool = True, max_depth: int = 3) -> Dict[str, int]:
        """Main method to scrape entire website"""
        session_id = self.start_scrape_session()
        urls_to_scrape = set()
        
        # Try to get URLs from sitemap first
        if use_sitemap:
            sitemap_urls = [f"{base_url}/sitemap.xml", f"{base_url}/sitemap_index.xml"]
            for sitemap_url in sitemap_urls:
                urls = self.get_sitemap_urls(sitemap_url)
                if urls:
                    urls_to_scrape.update(urls)
                    logger.info(f"Found {len(urls)} URLs in sitemap")
                    break
        
        # Discover URLs by crawling if no sitemap or if requested
        if not urls_to_scrape or discover_urls:
            discovered = self.discover_urls(base_url, max_depth)
            urls_to_scrape.update(discovered)
            logger.info(f"Discovered {len(discovered)} URLs by crawling")
        
        # Add base URL if not included
        urls_to_scrape.add(base_url)
        
        stats = {'scraped': 0, 'updated': 0, 'failed': 0}
        
        logger.info(f"Starting to scrape {len(urls_to_scrape)} URLs")
        
        for i, url in enumerate(urls_to_scrape, 1):
            logger.info(f"Scraping {i}/{len(urls_to_scrape)}: {url}")
            
            page_content = self.scrape_page(url)
            
            if page_content:
                if self.save_to_database(page_content):
                    stats['scraped'] += 1
                else:
                    stats['failed'] += 1
            else:
                stats['failed'] += 1
            
            # Rate limiting
            time.sleep(1)
        
        self.update_scrape_session(
            session_id, stats['scraped'], stats['updated'], stats['failed']
        )
        
        logger.info(f"Scraping completed: {stats}")
        return stats

# Usage example
if __name__ == "__main__":
    # Database configuration
    db_config = DB_CONFIG 
    
    
    scraper = WebsiteScraper(db_config)
    
    # Scrape website
    base_url = "https://technyder.co/"
    stats = scraper.scrape_website(
        base_url=base_url,
        use_sitemap=True,
        discover_urls=True,
        max_depth=3
    )
    
    print(f"Scraping completed: {stats}")


    #!/usr/bin/env python3
"""
Backfill embeddings for scraped_pages table
Only fills rows where embeddings_generated = false OR embeddings are null.
"""

# === Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_backfill")

genai.configure(api_key=GEMINI_API_KEY)
EMBED_MODEL = "models/text-embedding-004"


def embed_text(text: str):
    """Generate embedding using Gemini."""
    if not text:
        return None
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=text)
        return resp["embedding"]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def backfill_embeddings():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch rows missing embeddings
            cur.execute(f"""
                SELECT id, title, content
                FROM {TABLE_NAME}
                WHERE embeddings_generated = false
                   OR combined_embedding IS NULL
                LIMIT 50;
            """)
            rows = cur.fetchall()

            if not rows:
                logger.info("🎉 No rows need embedding generation!")
                return

            logger.info(f"Found {len(rows)} rows needing embeddings...")

            for row in rows:
                text_title = row["title"] or ""
                text_content = row["content"] or ""

                # Create embeddings
                title_emb = embed_text(text_title)
                content_emb = embed_text(text_content)
                combined_emb = embed_text(text_title + " " + text_content)

                if not combined_emb:
                    logger.warning(f"Skipping row {row['id']} (embedding failed)")
                    continue

                # Update DB
                cur.execute(f"""
                    UPDATE {TABLE_NAME}
                    SET title_embedding = %s,
                        content_embedding = %s,
                        combined_embedding = %s,
                        embeddings_generated = true,
                        embedding_created_at = NOW()
                    WHERE id = %s;
                """, (title_emb, content_emb, combined_emb, row["id"]))

                logger.info(f"✅ Updated embeddings for row {row['id']}")

            conn.commit()
            logger.info("🎯 Backfill complete!")


if __name__ == "__main__":
    backfill_embeddings()
