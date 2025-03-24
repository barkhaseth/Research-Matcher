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

chromedriver_path = 'C:/Users/misss/Downloads/chromedriver-win64/chromedriver.exe'
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

urls = [f'https://www.engineering.columbia.edu/faculty-staff/directory?search=&employee_type=&department=&strategic=&last_name=&pageindex={i}&pagesize=10&type=people' for i in range(0, 69)]

for link in urls:
    print(link)
    try:
        response = requests.head(link, allow_redirects=True)
        if response.status_code == 200:
            #print(f"Valid URL: {link}")

            driver.get(link)
            try:
                faculty_cards = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".card-faculty"))
                )

                for card in faculty_cards:
                    name = card.find_element(By.CSS_SELECTOR, ".card-faculty__heading").text
                    title = card.find_element(By.CSS_SELECTOR, ".card-faculty__title").text
                    profile_link = card.find_element(By.CSS_SELECTOR, "a.card-news__link").get_attribute('href')
                    img_src = card.find_element(By.CSS_SELECTOR, ".card-faculty__figure img").get_attribute('src')

                    print("Name:", name)
                    print("Title:", title)
                    print("Profile Link:", profile_link)
                    print("Image Link:", img_src)
                    print("---")

            except Exception as e:
                print(f"An error occurred while processing {link}: {e}")

                print("Full Page Source for Debugging:")
                print(driver.page_source)
        else:
            print(f"Invalid URL: {link} (Status code: {response.status_code})")

    except requests.exceptions.RequestException as e:
        print(f"Error checking URL {link}: {e}")

driver.quit()