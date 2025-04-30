# Scrape and Store

This project is a web scraping and data storage tool that crawls websites, extracts content, generates embeddings using OpenAI, and stores the data in a vector-enabled AstraDB collection.

## Features

- Scrapes content from a given URL.
- Extracts all links within the same domain or subdirectory.
- Generates embeddings for the scraped content using OpenAI's embedding model.
- Stores the content and embeddings in a vector-enabled AstraDB collection.

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mrnithesh/Scrape-nd-store.git
   cd Scrape-nd-store
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `ASTRA_DB_TOKEN`: Your AstraDB token.
   - `ASTRA_DB_URL`: Your AstraDB URL.

## Usage

1. Set the `start_url` in `main.py` to the website you want to scrape.
2. Run the script:
   ```bash
   python main.py
   ```

## Notes

- Ensure that the AstraDB collection is vector-enabled with the correct schema.
- The embedding model used is `text-embedding-3-small`. Update it as needed.

## License

This project is licensed under the MIT License.