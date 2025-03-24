import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
from scholarly import scholarly

# chromedriver_path = 'C:/Users/misss/Downloads/chromedriver-win64/chromedriver.exe'
# service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

def get_gs_info(fac_json, output_file):
    for prof_id, prof_info in fac_json.items():
        if "google-scholar" in prof_info:
            print(prof_info["google-scholar"])
            
            url = prof_info["google-scholar"]
            if url=="https://scholar.google.com/citations?user=":
                print("skipped")
                continue
            try:
                start_index = url.find('user=') + len('user=')
                author_id = url[start_index:]
                print(f"Extracted Author ID: {author_id}")

                if not author_id:
                    print("No author ID found, skipping...")
                    continue
                
                author = scholarly.search_author_id(author_id)

                # Fill the author info
                author = scholarly.fill(author)  

                articles = []
                for pub in author['publications']:
                    title = pub['bib'].get('title', 'No Title Available')  # Safe retrieval
                    year = pub['bib'].get('pub_year', 'Unknown')
                    article_link = pub.get('pub_url', "No link available")  # Retrieve article link

                    if year != 'Unknown':
                        articles.append({
                            'title': title,
                            'year': int(year) if year.isdigit() else year
                        })

                sorted_articles = sorted(articles, key=lambda x: x["year"], reverse=True)

                prof_info['sorted_articles'] = sorted_articles
                
                for article in sorted_articles:
                    print(f"{article['year']}: {article['title']} - {article['link']}")

            except requests.exceptions.RequestException as e:
                continue
            except Exception as e:
                continue

    with open(output_file, 'w') as json_file:
        json.dump(fac_json, json_file, indent=4)

                
    
def main():
    fac_with_scholar = "C:/Users/misss/Downloads/chromedriver-win64/fac_info_with_scholar.json"
    with open(fac_with_scholar, 'r') as file:
        data = json.load(file)
    
    
    output_file = 'C:/Users/misss/Downloads/chromedriver-win64/faculty_info_with_articles.json'
    get_gs_info(data, output_file)
        
if __name__ == "__main__":
    main()