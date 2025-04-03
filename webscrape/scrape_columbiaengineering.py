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

def create_service():
	service = Service(ChromeDriverManager().install())

def scrape_faculty_data(service, options):

	num_pages = 24
	# employee_type = 61 specifies Faculty only cards
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

				## Opening a new driver to examine profile link ##
				profile_driver = webdriver.Chrome(service=create_service(), options=options)
				profile_driver.get(profile_link)

				 ## Retrieve Email ##
				try:
					WebDriverWait(profile_driver, 10).until(
						EC.presence_of_element_located((By.CSS_SELECTOR, "nav.rail-contact"))
					)
					email_elem = profile_driver.find_element(By.CSS_SELECTOR, ".rail-contact__email a")
					email = email_elem.get_attribute("href").replace("mailto:", "")
				except:
					email = "N/A"

				## Retrieve Additional Links ##
				try:
					cta_list = profile_driver.find_element(By.CSS_SELECTOR, "ul.rail-cta__list").find_elements(By.CSS_SELECTOR, "li.rail-cta__item")

					for item in cta_list:
						label_link = item.find_element(By.TAG_NAME, "a").get_attribute("href")
						label = item.find_element(By.CLASS_NAME, "text").text
						label = label.replace(" ", "-").lower()

						# print(label, label_link)
						additional_links[label] = label_link

				except:
					additional_links = {}

					## Retrieve Intro and Research Areas ##
					research_intro = ""
					research_areas = []

					try:
						WebDriverWait(profile_driver, 10).until(
							EC.presence_of_element_located((By.CSS_SELECTOR, "div.intro"))
						)

						intro = profile_driver.find_element(By.CSS_SELECTOR, "div.intro")

						intro_text = intro.find_element(By.TAG_NAME, "p").text
						research_intro += intro_text
					except:
						pass

					try:
						sections = profile_driver.find_elements(By.CSS_SELECTOR, "section.wysiwyg.component")
						for sec in sections:
							try:
								## Research Areas ##
								sec_head = sec.find_elements(By.TAG_NAME, "h2")[0].text
								if sec_head == "Research Areas":
									research_list = sec.find_elements(By.TAG_NAME, "li")
									for area in research_list:
										research_areas.append(area.text)
							except:
								## Intro Paragraphs ##
								paragraphs = sec.find_elements(By.TAG_NAME, "p")
								for p in paragraphs:
									research_intro += p.text.replace("\n", " ")
					except:
						pass

				profile_driver.quit()

				## Adding to JSON ##
				faculty_data[name] = {
					"name": name,
					"title": title,
					"profile-link": profile_link,
					"intro": research_intro,
					"email": email,
					"img": img_src,
					"research-areas": research_areas
				}

				for label in additional_links:
					faculty_data[name][label] = additional_links[label]

				## For debugging use ##
				# print_card_info(name, title, profile_link, img_src)

		except Exception as e:
			print(f"An error occurred while processing {link}: {e}")
			print("Full Page Source for Debugging:")
			print(driver.page_source)

		driver.quit()

	print("Total Faculty Scraped:", total_faculty)
	return faculty_data

def json_dump(faculty_data, path):
	with open(path, "w") as file:
		json.dump(faculty_data, file, indent=4)

## unused ##
def scrape_profiles(service, options, profile_urls):

	for i, link in enumerate(profile_urls):
		driver = webdriver.Chrome(service=service, options=options)
		driver.minimize_window()

		try:
			driver.get(link)

		except Exception as e:
			print(f"An error occurred while processing {link}: {e}")
			print("Full Page Source for Debugging:")
			print(driver.page_source)

		driver.quit()

def main():
	service = create_service()

	options = Options()
	options.add_argument("--window-size=2,2")  # Width x Height

	scraped_data = scrape_faculty_data(service, options)

	path = "faculty.json"
	json_dump(scraped_data, path)

if __name__ == "__main__":
	main()
