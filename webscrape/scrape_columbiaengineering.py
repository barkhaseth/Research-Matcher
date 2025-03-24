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

def print_card_info(name, title, profile_link, img_src):
    print("Name:", name)
    print("Title:", title)
    print("Profile Link:", profile_link)
    print("Image Link:", img_src)
    print("---")

def scrape_faculty_data():
    service = Service(ChromeDriverManager().install())

    options = Options()
    options.add_argument("--window-size=2,2")  # Width x Height
    options.add_argument("--start-minimized")

    num_pages = 24
    # employee_type = 61 specifies Faculty only
    urls = [f'https://www.engineering.columbia.edu/faculty-staff/directory?search=&employee_type=61&department=&strategic=&last_name=&pageindex={i}&pagesize=12&type=people' for i in range(0, num_pages)]

    total_faculty = 0
    faculty_data = {}

    for i, link in enumerate(urls):
        print("Scraping page", i + 1)

        driver = webdriver.Chrome(service=service, options=options)
        driver.minimize_window()

        try:
            driver.get(link)
            faculty_cards = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".card-faculty"))
            )
            total_faculty += len(faculty_cards)

            for card in faculty_cards:
                name = card.find_element(By.CSS_SELECTOR, ".card-faculty__heading").text
                title = card.find_element(By.CSS_SELECTOR, ".card-faculty__title").text
                profile_link = card.find_element(By.CSS_SELECTOR, "a.card-news__link").get_attribute('href')
                img_src = card.find_element(By.CSS_SELECTOR, ".card-faculty__figure img").get_attribute('src')

                faculty_data[name] = {
                    "name": name,
                    "title": title,
                    "profile-link": profile_link,
                    "img": img_src,
                }

                ## For debugging use ##
                # print_card_info(name, title, profile_link, img_src)

        except Exception as e:
            print(f"An error occurred while processing {link}: {e}")
            print("Full Page Source for Debugging:")
            print(driver.page_source)

        driver.quit()

    print("Total Faculty Scraped:", total_faculty)
    return faculty_data

def json_dump(faculty_data):
    with open("faculty.json", "w") as file:
        json.dump(faculty_data, file, indent=4)

def main():
    scraped_data = scrape_faculty_data()
    json_dump(scraped_data)

if __name__ == "__main__":
    main()
