from sentence_transformers import SentenceTransformer

def main():
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
	proposal = """medical image machine learning"""

	proposal_embedding = model.encode(proposal)

if __name__ == '__main__':
	main()