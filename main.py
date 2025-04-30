import requests
from bs4 import BeautifulSoup
import openai
import os
from urllib.parse import urljoin, urlparse
from astrapy import DataAPIClient

# Set your OpenAI API key and AstraDB credentials
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")  # Replace with your AstraDB token
ASTRA_DB_URL = os.getenv("ASTRA_DB_URL")  # Replace with your AstraDB URL

# Initialize AstraDB client
client = DataAPIClient(ASTRA_DB_TOKEN)
db = client.get_database_by_api_endpoint(ASTRA_DB_URL)

# Function to scrape content from a given URL
def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
            print(f"Crawling {url}... Success")
            return text_content
        else:
            print(f"Failed to retrieve {url} - Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error while scraping {url}: {e}")
        return None

# Function to extract all links from the same domain/subdirectory
def extract_links(url, base_domain):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract both relative and absolute URLs
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                absolute_url = urljoin(url, href)  # Convert relative URL to absolute URL
                
                # Only keep URLs within the same base domain
                if urlparse(absolute_url).netloc == base_domain:
                    links.add(absolute_url)

            return links
        else:
            print(f"Failed to retrieve {url} - Status code: {response.status_code}")
            return set()
    except Exception as e:
        print(f"Error extracting links from {url}: {e}")
        return set()

# Generate embeddings using OpenAI's embedding model
def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to create a vector-enabled collection if it doesn't exist
def create_vector_enabled_collection_if_not_exists(collection_name):
    try:
        # List existing collections in AstraDB
        collections = db.list_collection_names()

        # Create collection if it doesn't exist
        if collection_name not in collections:
            print(f"Collection '{collection_name}' does not exist. Creating it now...")

            # Vector-enabled schema with vector support
            schema = {
                "fields": {
                    "url": "string",
                    "content": "string",
                    "embedding_vector": {
                        "type": "vector<float>",  # Vector field type for embeddings
                        "dimension": 1536         # Define dimension of embedding vector
                    }
                }
            }

            db.create_collection(collection_name, schema=schema)
            print(f"Vector-enabled collection '{collection_name}' created successfully.")
        return True
    except Exception as e:
        print(f"Error while creating collection '{collection_name}': {e}")
        return False
# Function to split the embedding into smaller chunks
def split_embedding(embedding, chunk_size=1000):
    return [embedding[i:i + chunk_size] for i in range(0, len(embedding), chunk_size)]

# Store the embeddings in AstraDB (vector-enabled collection)
def store_in_astra_vector_collection(url, content, embedding):
    try:
        collection_name = "collection_name"  # Replace with your desired collection name
        
        # Check if the collection exists or create it
        if not create_vector_enabled_collection_if_not_exists(collection_name):
            print(f"Cannot proceed without collection '{collection_name}'.")
            return
        
        collection = db.get_collection(collection_name)

        # Split embedding into smaller chunks
        embedding_chunks = split_embedding(embedding, chunk_size=1000)
        
        document = {
            "url": url,
            "content": content,
        }
        
        # Store each chunk in a separate field
        for index, chunk in enumerate(embedding_chunks):
            document[f"embedding_{index + 1}"] = chunk  # Save chunks as embedding_1, embedding_2, etc.

        # Insert the document
        collection.insert_one(document)
        print(f"Successfully stored content from {url} in collection {collection_name}")
    except Exception as e:
        print(f"Error storing data in AstraDB: {e}")

# Main function to crawl and store all data from subdirectories of the base URL
def crawl_and_store(start_url):
    visited_urls = set()
    urls_to_visit = set([start_url])
    base_domain = urlparse(start_url).netloc  # Extract the base domain to restrict crawling

    while urls_to_visit:
        current_url = urls_to_visit.pop()

        if current_url not in visited_urls:
            visited_urls.add(current_url)

            # Scrape the website
            content = scrape_website(current_url)
            if content:
                # Generate embeddings for the content
                embedding = generate_embedding(content)
                if embedding:
                    # Store the scraped data and embedding in the vector-enabled AstraDB collection
                    store_in_astra_vector_collection(current_url, content, embedding)

            # Extract links and add them to the queue (only links in the same base domain)
            new_links = extract_links(current_url, base_domain)
            urls_to_visit.update(new_links - visited_urls)

# Start crawling from the main page
if __name__ == "__main__":
    start_url = ""  # Set the website you want to start scraping
    crawl_and_store(start_url)
