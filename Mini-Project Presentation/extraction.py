from bs4 import BeautifulSoup
import requests
import json

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

    # Extract introduction and body paragraphs, clean content
    introduction = paragraphs[0].get_text(strip=True).replace("\n", "") if paragraphs else ""
    body = [
        p.get_text(strip=True).replace("\n", "").encode().decode('unicode_escape')
        for p in paragraphs[1:] if p.get_text(strip=True)
    ]

    # Prepare the structured JSON
    output_data = {
        "title": title,
        "context": "This JSON contains an article about the Internet. The information is derived from Wikipedia and has been cleaned to remove tables, references, newline characters, and encoded Unicode sequences.",
        "article": {
            "introduction": introduction.encode().decode('unicode_escape'),
            "body": body
        }
    }

    # Save the JSON to a file
    with open("internet_article_llm_cleaned.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print("Data saved to internet_article_llm_cleaned.json")
else:
    print("Failed to retrieve the webpage")
