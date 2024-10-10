import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BASE_URL = "https://example.com"  # Replace with the website you want to scrape
CHUNK_SIZE = 1000  # Adjust as needed
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
OPENAI_API_KEY = "your-openai-api-key"  # Replace with your actual API key
QDRANT_URL = "http://localhost:6333"  # Replace with your Qdrant server URL
COLLECTION_NAME = "web_content"

openai.api_key = OPENAI_API_KEY

class WebScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()

    def scrape_website(self) -> List[Dict[str, Any]]:
        """
        Scrape the entire website starting from the base URL.
        """
        to_visit = [self.base_url]
        scraped_data = []

        while to_visit:
            url = to_visit.pop(0)
            if url in self.visited_urls:
                continue

            logging.info(f"Scraping: {url}")
            content, new_urls = self._scrape_page(url)
            if content:
                scraped_data.append({"url": url, "content": content})
                self.visited_urls.add(url)
                to_visit.extend(new_urls)

        return scraped_data

    def _scrape_page(self, url: str) -> tuple:
        """
        Scrape a single page and return its content and new URLs to visit.
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
                new_urls = self._extract_urls(soup)
                return content, new_urls
            except requests.RequestException as e:
                logging.warning(f"Error scraping {url}: {e}. Attempt {attempt + 1} of {MAX_RETRIES}")
                if attempt == MAX_RETRIES - 1:
                    logging.error(f"Failed to scrape {url} after {MAX_RETRIES} attempts")
                    return None, []
                time.sleep(RETRY_DELAY)

    def _extract_urls(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract and normalize URLs from a BeautifulSoup object.
        """
        urls = []
        for link in soup.find_all('a', href=True):
            url = urljoin(self.base_url, link['href'])
            if self._is_valid_url(url):
                urls.append(url)
        return urls

    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and belongs to the same domain as the base URL.
        """
        parsed_url = urlparse(url)
        parsed_base = urlparse(self.base_url)
        return parsed_url.netloc == parsed_base.netloc and parsed_url.scheme in ('http', 'https')

class TextProcessor:
    @staticmethod
    def chunk_text(text: str, chunk_size: int) -> List[str]:
        """
        Split the text into chunks of approximately chunk_size tokens.
        This is a simple implementation and can be improved for better semantic splitting.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += 1
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

class Vectorizer:
    @staticmethod
    def generate_embeddings(texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks using OpenAI's API.
        """
        embeddings = []
        for i in range(0, len(texts), 100):  # Process in batches of 100
            batch = texts[i:i+100]
            try:
                response = openai.Embedding.create(input=batch, model="text-embedding-ada-002")
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"Error generating embeddings: {e}")
                raise

        return embeddings

class QdrantManager:
    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url)
        self.collection_name = collection_name

    def create_collection_if_not_exists(self, vector_size: int):
        """
        Create a new collection if it doesn't already exist.
        """
        collections = self.client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logging.info(f"Created new collection: {self.collection_name}")
        else:
            logging.info(f"Collection {self.collection_name} already exists")

    def upsert_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """
        Upsert vectors and their associated payloads into the Qdrant collection.
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=[i for i in range(len(vectors))],
                    vectors=vectors,
                    payloads=payloads
                )
            )
            logging.info(f"Upserted {len(vectors)} vectors into {self.collection_name}")
        except Exception as e:
            logging.error(f"Error upserting vectors: {e}")
            raise

def main():
    # Step 1: Web Scraping
    scraper = WebScraper(BASE_URL)
    scraped_data = scraper.scrape_website()

    # Step 2: Chunking the Data
    text_processor = TextProcessor()
    chunked_data = []
    for item in scraped_data:
        chunks = text_processor.chunk_text(item['content'], CHUNK_SIZE)
        for chunk in chunks:
            chunked_data.append({
                "text": chunk,
                "url": item['url']
            })

    # Step 3: Vectorization
    vectorizer = Vectorizer()
    texts = [item['text'] for item in chunked_data]
    vectors = vectorizer.generate_embeddings(texts)

    # Step 4: Storing in Qdrant
    qdrant_manager = QdrantManager(QDRANT_URL, COLLECTION_NAME)
    qdrant_manager.create_collection_if_not_exists(len(vectors[0]))
    
    payloads = [{"text": item['text'], "url": item['url']} for item in chunked_data]
    qdrant_manager.upsert_vectors(vectors, payloads)

    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()