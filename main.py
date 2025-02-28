from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from openai import OpenAI
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from waitress import serve
from pinecone import Pinecone
import subprocess

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv('PINECONE_INDEX_NAME')
if not pc.has_index(index_name):
    subprocess.run(["python", "pinecone_setup.py"])

app = Flask(__name__)
    
#NEW
with open('faculty_info_uniquekeywords.json', 'r') as json_file:
    faculty_keywords = json.load(json_file)
    
#df = pd.DataFrame(loaded_data_list)

#NEW
#keywords_df = pd.DataFrame(faculty_keywords)
keywords_df = pd.DataFrame.from_dict(faculty_keywords, orient='index')
keywords_df.reset_index(inplace=True)
keywords_df.rename(columns={'index': 'id'}, inplace=True)

# Initialize the sentence embedder model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cpu')

logging.basicConfig(filename='app_log.txt', level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    global keywords_df
    generate_title_checkbox_checked = False

    if request.method == 'POST':
        
        log_entry = {'text_content': request.form['proposal'], 'timestamp': str(datetime.now())}
        logging.info(json.dumps(log_entry))

        proposal = request.form['proposal']
        result_count = int(request.form['resultCount'])
        generate_title_checkbox = 'generateTitle' in request.form
        generate_title_checkbox_checked = generate_title_checkbox

        # Assuming 'proposal_embedding' contains the 1D array of the embedded proposal vector
        proposal_embedding = model.encode([proposal])[0]

        keywords_df['Name_lower'] = keywords_df['name'].str.lower()

        # NEW: Query Pinecone
        query_results = query_pinecone(proposal_embedding, top_k=result_count)
        results = []
        for match in query_results["matches"]:
            metadata = match["metadata"]
            name_lower = metadata["name"].lower()

            keywords = keywords_df.loc[keywords_df['Name_lower'] == name_lower, 'keywords'].values
            keywords = keywords[0] if len(keywords) > 0 else None

            if not keywords:
                keywords = []
            elif isinstance(keywords, str):
                keywords = [keyword.strip() for keyword in keywords.split(',')]
            elif not isinstance(keywords, list):
                keywords = []  # Default to empty list if it's not already a list


            results.append({
                "Name": metadata["name"].title(),
                "Title": metadata["title"].title(),
                "Research Summary": metadata["researchSummary"],
                "One-Line Summary": generate_one_line_summary(metadata["researchSummary"], proposal),
                "keywords": keywords,
                "email_body": generate_email_body(proposal, metadata["researchSummary"])
            })
        
        generated_title = None

        if generate_title_checkbox:
            # Generate a title using OpenAI chat-based language model
            generated_title = generate_title(proposal)

        # Render the template
        return render_template('index.html', proposal=proposal, results=results, resultCount=result_count, generatedTitle=generated_title, generateTitleCheckbox=generate_title_checkbox_checked)

    # For GET requests, set the checkbox to unchecked by default
    return render_template('index.html', generateTitleCheckbox=generate_title_checkbox_checked)

def generate_title(proposal):
    # Call the OpenAI API to generate a title using chat-based language model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that generates a simple title for a research paper given a research proposal. Do not include the word title in the response."},
            {"role": "user", "content": proposal}
        ],
        temperature=0.5,
        max_tokens=50,
    )

    return response.choices[0].message.content.strip()

def generate_one_line_summary(research_summary, proposal):
    # Call the OpenAI API to summarize the research summary into one relevant line
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes long research summaries into a single line. The summary should be relevant to the given proposal."},
            {"role": "user", "content": f"Proposal: {proposal}\nResearch Summary: {research_summary}"}
        ],
        temperature=0.7,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()

def query_pinecone(proposal_embedding, top_k):
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
    try:
        results = index.query(
            vector=proposal_embedding.tolist(),
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"matches": []} 
    
def generate_email_body(proposal, faculty_description):
    prompt = f"""
    Write a professional email combining the following research proposal and faculty project description.
    Use the proposal as the starting point and the faculty's description as additional context. The email should be polite and professional.

    Research Proposal:
    {proposal}

    Faculty Project Description:
    {faculty_description}
    """

    response = client.chat.completions.create(
       model="gpt-4o",  
        messages=[
           {"role": "system", "content": "You are an assistant that generates professional emails."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    # serve(app, host='0.0.0.0')
