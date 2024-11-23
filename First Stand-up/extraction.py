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

    # Extract all paragraphs in the article
    paragraphs = soup.find_all('p')

    # Combine paragraphs into a single text block
    content = "\n\n".join(p.get_text() for p in paragraphs)

    # Combine title and content into plain text
    plain_text = f"Title: {title}\n\n{content}"

    # Prepare the JSON data in chunks without breaking sentences
    json_data_chunks = text_to_json_chunks(title, plain_text, chunk_size=2000)

    # Save all chunks to a single JSON file
    output_file = "internet_article_chunks_combined.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(json_data_chunks, json_file, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_file}")
    print(f"Length of the text: {len(plain_text)} characters")
else:
    print("Failed to retrieve the webpage")
