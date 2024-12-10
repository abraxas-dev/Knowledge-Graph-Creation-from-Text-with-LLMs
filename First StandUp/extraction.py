import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# Download punkt tokenizer
nltk.download('punkt')

# Function to fetch webpage content
def fetch_webpage(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to retrieve the webpage: {response.status_code}")

# Function to clean and parse HTML content
def parse_and_clean_html(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove all tables
    for table in soup.find_all('table'):
        table.decompose()

    # Remove all references
    for ref in soup.find_all('sup', {'class': 'reference'}):
        ref.decompose()
    
    return soup

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=2000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Function to save sections to files
def save_sections_to_files(sections, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for section_title, section_content in sections.items():
        if section_content.strip():
            section_filename = os.path.join(output_directory, f"{section_title}.txt")
            with open(section_filename, "w", encoding="utf-8") as section_file:
                section_file.write(section_content.strip())

# Main function to extract and save content from a webpage
def main(url, output_directory):
    try:
        html_content = fetch_webpage(url)
        soup = parse_and_clean_html(html_content)
        
        # Extract the title
        title = soup.title.string.strip()

        # Extract sections
        elements = soup.find_all(['h2', 'h3', 'p'])
        sections = {}
        current_section_title = "Introduction"
        current_section_content = ""

        for element in elements:
            if element.name in ['h2', 'h3']:
                if current_section_content.strip():
                    sections[current_section_title] = current_section_content
                current_section_title = element.get_text().strip().replace("/", "-")
                current_section_content = ""
            elif element.name == 'p':
                current_section_content += element.get_text().strip() + "\n\n"
        
        if current_section_content.strip():
            sections[current_section_title] = current_section_content

        # Save sections to files
        save_sections_to_files(sections, output_directory)
        print(f"Sections saved to the directory: {output_directory}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    URL = "https://en.wikipedia.org/wiki/Internet"
    OUTPUT_DIRECTORY = "internet_article_sections"
    main(URL, OUTPUT_DIRECTORY)
