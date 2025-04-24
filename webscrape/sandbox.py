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
from sentence_transformers import SentenceTransformer
import json
import time

def main():
	input_file = "../data/faculty_complete.json.old"

	output_file = "../data/faculty_complete.json"

	with open(input_file, "r") as json_file:
		data = json.load(json_file)

	new_data = {}
	for key, data in data.items():
		new_key = key.lower().replace(" ", "-")
		print(new_key)
		new_data[new_key] = data

	with open(output_file, 'w') as output:
		json.dump(new_data, output)

if __name__ == "__main__":
	main()