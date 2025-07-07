import requests
from bs4 import BeautifulSoup
import json
import time
from scholarly import scholarly

def get_gs_info(fac_json, output_file):
	for prof_id, prof_info in fac_json.items():
		if "google-scholar" in prof_info:
			print(prof_info["google-scholar"])

			url = prof_info["google-scholar"]
			if url == "https://scholar.google.com/citations?user=":
				print("skipped")
				continue
			try:
				author_id = url.split("user=")[1].split("&")[0]
				print(f"Extracted Author ID: {author_id} for {prof_id}")

				if not author_id:
					print("No author ID found, skipping...")
					continue

				try:
					author = scholarly.search_author_id(author_id)
				except:
					continue

				author = scholarly.fill(author)

				articles = []
				article_flag = False

				for pub in author['publications']:
					title = pub['bib'].get('title', 'No Title Available')
					year = pub['bib'].get('pub_year', 'Unknown')

					pub = scholarly.fill(pub, ['pub_url'])
					article_link = pub.get('pub_url', "No link available")
					if article_link != "No link available":
						article_flag = True

					if year != 'Unknown':
						articles.append({
							'title': title,
							'year': int(year) if year.isdigit() else year,
							'link': article_link
						})

				if article_flag:
					print("retrieved articles successfully")

				sorted_articles = sorted(articles, key=lambda x: x["year"], reverse=True)
				prof_info['sorted-articles'] = sorted_articles

				# Upsert immediately
				with open(output_file, 'w') as json_file:
					json.dump(fac_json, json_file, indent=4)

			except requests.exceptions.RequestException:
				print("Scholar not found")
				continue
			except Exception:
				print("Scholar not found")
				continue


def scrape_scholar():
	print("Scraping from Google Scholars")
	fac =  "../data/faculty.json"
	with open(fac, 'r') as file:
		data = json.load(file)

	output_file =  "../data/faculty_complete.json"
	get_gs_info(data, output_file)

def main():
	scrape_scholar()

if __name__ == "__main__":
	main()