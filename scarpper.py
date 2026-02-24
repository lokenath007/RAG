import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://www.w3schools.com"

def scrape_page(url):
    print("Scraping:", url)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # Remove nav/footer/script/style
    for tag in soup(["script","style","nav","footer","header"]):
        tag.decompose()
    text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
    return text

def get_links(topic_url):
    r = requests.get(topic_url)
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(topic_url.replace(BASE_URL,"")) or href.startswith("/"+topic_url.split("/")[-2]):
            links.append(BASE_URL + href)
    return list(set(links))

# Example: Scrape all Python pages
topic_url = f"{BASE_URL}/python/"
pages = get_links(topic_url)

all_text = []
for url in pages:
    all_text.append(scrape_page(url))
    time.sleep(1)  # polite scraping