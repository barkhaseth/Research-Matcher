from sentence_transformers import SentenceTransformer
import json

def iterate_faculty(faculty_data, model):

	research_areas = ""
	intro = ""
	article_titles = ""

	for prof_key, data in faculty_data.items():
		if "intro" in data and len(data['intro']) > 0:
			intro = "Intro: " + data['intro'] + " "

		if "research-areas" in data and len(data['research-areas']) > 0:
			research_areas = "Research Areas: " + ", ".join(data['research-areas']) + " "

		if "sorted_articles" in data:
			titles = [article["title"] for article in data["sorted_articles"]]
			article_titles = "Article Titles: " + " ".join([title if title.endswith('.') else title + '.' for title in titles])

		final = intro + research_areas + article_titles
		if len(final) == 0:
			final = data['title']

		embedding = model.encode(final.strip())
		faculty_data[prof_key]["embedding"] = embedding.tolist()

	return data

def main():
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

	data_file = "./data/faculty_complete.json"

	with open(data_file, "r") as file:
		data = json.load(file)

	output_data = iterate_faculty(data, model)
	# print(output_data)

	output_file = "./data/faculty_complete_embeddings.json"

	with open(output_file, 'w') as o:
		json.dump(output_data, o)

	# Research Areas, Intro, Article Titles

if __name__ == '__main__':
	main()