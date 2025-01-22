import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from logger_config import setup_logger

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")


class Extractor:
    
    def __init__(self, urls, processed_data_path, chunk_size=1500):
        """
        Initialize the extractor with URLs, output path, and chunk size.

        Args:
            urls (list): List of URLs to process.
            processed_data_path (str): Path to save processed data.
            chunk_size (int): Maximum size of each text chunk.
        """
        self.logger = setup_logger(__name__)
        self.urls = urls  # List of URLs to process
        self.processed_data_path = processed_data_path
        self.chunk_size = chunk_size

        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
            self.logger.info(f"✅ Created output directory: {processed_data_path}")

    def fetch_webpage(self, url):
        """Fetch webpage content."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.logger.info(f"✅ Successfully fetched content from {url}")
            return response.content
        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to fetch URL {url}: {e}")
            return None

    def parse_and_clean_html(self, html_content):
        """Clean and parse HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'lxml')

            # Remove tables and references
            tables_removed = len(soup.find_all('table'))
            refs_removed = len(soup.find_all('sup', {'class': 'reference'}))
            
            for table in soup.find_all('table'):
                table.decompose()
            for ref in soup.find_all('sup', {'class': 'reference'}):
                ref.decompose()

            self.logger.info(f"🧹 Cleaned HTML content (removed {tables_removed} tables and {refs_removed} references)")
            return soup
        except Exception as e:
            self.logger.error(f"❌ Error parsing HTML content: {e}")
            raise

    def split_text_into_chunks(self, text):
        """Split text into chunks."""
        try:
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

            self.logger.info(f"📄 Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"❌ Error splitting text into chunks: {e}")
            raise

    def save_chunks_to_files(self, chunks, output_directory, base_name="chunk"):
        """Save text chunks into files."""
        try:
            for idx, chunk in enumerate(chunks):
                filename = os.path.join(output_directory, f"{base_name}_{idx + 1}.txt")
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(chunk.strip())
            
            self.logger.info(f"✅ Saved {len(chunks)} chunks to {output_directory}")
        except Exception as e:
            self.logger.error(f"❌ Error saving chunks to files: {e}")
            raise

    def preprocess(self):
        """Main method to process all URLs."""
        self.logger.info("="*50)
        self.logger.info("🔄 Starting Content Extraction")
        self.logger.info("="*50)

        for url in self.urls:
            try:
                self.logger.info(f"📝 Processing URL: {url}")

                # Fetch webpage content
                html_content = self.fetch_webpage(url)
                if not html_content:
                    continue  # Skip to the next URL if fetching fails

                # Parse and clean HTML
                soup = self.parse_and_clean_html(html_content)

                # Extract plain text from cleaned HTML
                text = " ".join([p.get_text() for p in soup.find_all('p')])
                chunks = self.split_text_into_chunks(text)

                # Create a folder based on the page title
                page_title = soup.title.string.strip().replace(" ", "_") if soup.title else "unknown_page"
                output_directory = os.path.join(self.processed_data_path, page_title)
                os.makedirs(output_directory, exist_ok=True)

                # Save the chunks in the specific folder
                self.save_chunks_to_files(chunks, output_directory)
                self.logger.info(f"✅ Successfully processed content from {url}")
                self.logger.info(f"📁 Saved in: {output_directory}")

            except Exception as e:
                self.logger.error(f"❌ Error processing URL {url}: {e}")
                continue

        self.logger.info("="*50)
        self.logger.info("✅ Content Extraction Completed")
        self.logger.info("="*50)


if __name__ == "__main__":
    # Sample test configuration
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        
    ]
    output_path = "./test_processed_data"
    chunk_size = 500

    # Create and run the extractor
    extractor = Extractor(test_urls, output_path, chunk_size)
    extractor.preprocess()
