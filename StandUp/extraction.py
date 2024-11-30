from bs4 import BeautifulSoup
import requests
import json
import os
import nltk

# Download punkt tokenizer for sentence splitting
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Internet"

# Fetch the content of the page
response = requests.get(url)

# Function to split text into chunks with approximately 2000 characters, without breaking sentences
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

# Function to prepare JSON format for chunks
def text_to_json_chunks(title, text, chunk_size=2000):
    chunks = split_text_into_chunks(text, chunk_size)
    return [{"title": title, "chunk_index": i + 1, "content": chunk, "character_count": len(chunk)} for i, chunk in enumerate(chunks)]

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'lxml')

    # Remove all tables
    for table in soup.find_all('table'):
        table.decompose()

    # Remove all references (superscript citation links)
    for ref in soup.find_all('sup', {'class': 'reference'}):
        ref.decompose()

    # Extract the article title
    title = soup.title.string.strip()

    # Extract all headings and paragraphs in the article
    sections = soup.find_all(['h2', 'h3', 'p'])

    # Directory to save text files
    output_directory = "internet_article_sections"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    current_section_title = "Introduction"
    current_section_content = ""

    # Iterate through all sections to separate content under headings
    for element in sections:
        if element.name in ['h2', 'h3']:
            # Save the previous section content to a text file
            if current_section_content.strip():
                section_filename = os.path.join(output_directory, f"{current_section_title}.txt")
                with open(section_filename, "w", encoding="utf-8") as section_file:
                    section_file.write(current_section_content.strip())

            # Update the section title and reset content
            current_section_title = element.get_text().strip().replace("/", "-")
            current_section_content = ""
        elif element.name == 'p':
            # Append paragraph content to the current section
            current_section_content += element.get_text().strip() + "\n\n"

    # Save the last section content to a text file
    if current_section_content.strip():
        section_filename = os.path.join(output_directory, f"{current_section_title}.txt")
        with open(section_filename, "w", encoding="utf-8") as section_file:
            section_file.write(current_section_content.strip())

    print(f"Sections saved to the directory: {output_directory}")
else:
    print("Failed to retrieve the webpage")
