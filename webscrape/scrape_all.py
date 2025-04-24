from scrape_scholar import scrape_scholar
from scrape_columbiaengineering import scrape_columbia

"""
Scrapes data about all Columbia engineering faculty from scratch
	1. scrape_columbia() gathers all faculty info from Columbia Engineering profile link
		and dumps it in "data/faculty.json"
	2. scrape_scholar() finds + opens Google Scholar links from "data/faculty.json" and
		adds articles to professor jsons, then dumps in "data/faculty_complete.json"
"""
def main():
	scrape_columbia()
	scrape_scholar()


if __name__ == "__main__":
	main()