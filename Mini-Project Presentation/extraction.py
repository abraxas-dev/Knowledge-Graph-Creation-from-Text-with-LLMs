from bs4 import BeautifulSoup
import requests

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Internet"

# Fetch the content of the page
response = requests.get(url)

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

    # Save the plain text to a file
    with open("internet_article_raw.txt", "w", encoding="utf-8") as f:
        f.write(plain_text)

    # Calculate the length of the text
    text_length = len(plain_text)
    print(f"Data saved to internet_article_raw1.txt")
    print(f"Length of the text: {text_length} characters")
else:
    print("Failed to retrieve the webpage")
