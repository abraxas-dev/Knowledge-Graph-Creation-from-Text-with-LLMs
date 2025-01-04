import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


class Extractor:
    def __init__(self, urls, processed_data_path, chunk_size=1500):
        self.urls = urls  # List of URLs from config
        self.processed_data_path = processed_data_path
        self.chunk_size = chunk_size

        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)

    def fetch_webpage(self, url):
        """Fetch webpage content."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Failed to fetch URL {url}: {e}")
            return None

    def parse_and_clean_html(self, html_content):
        """Clean and parse HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')

        for table in soup.find_all('table'):
            table.decompose()
        for ref in soup.find_all('sup', {'class': 'reference'}):
            ref.decompose()

        return soup

    def split_text_into_chunks(self, text):
        """Split text into chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def save_chunks_to_files(self, chunks, output_directory, base_name="chunk"):
        """Save text chunks into files."""
        for idx, chunk in enumerate(chunks):
            filename = os.path.join(output_directory, f"{base_name}_{idx + 1}.txt")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(chunk.strip())

    def preprocess(self):
        """Main method to process all URLs."""
        for url in self.urls:
            print(f"Processing URL: {url}")

            html_content = self.fetch_webpage(url)
            if not html_content:
                continue

            soup = self.parse_and_clean_html(html_content)

            # Extract plain text from cleaned HTML
            text = " ".join([p.get_text() for p in soup.find_all('p')])
            chunks = self.split_text_into_chunks(text)

            # Use domain name as directory name
            domain_name = url.split("//")[-1].split("/")[0]
            output_directory = os.path.join(self.processed_data_path, domain_name)
            os.makedirs(output_directory, exist_ok=True)

            self.save_chunks_to_files(chunks, output_directory)
            print(f"Processed and saved content for {url} in {output_directory}")
