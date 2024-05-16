from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import torch
import pickle
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn.functional as F
import itertools
import math
import pickle
import fitz  # PyMuPDF
###
import spacy
from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor
import spacy
from spacy.matcher import Matcher
import PyPDF2
import csv
# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
############################
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

import torch

# Load the job embeddings
with open('job_embeddings.pickle', 'rb') as f:
    job_embeddings = pickle.load(f)

job_embeddings = list(job_embeddings.values())
job_embeddings = torch.tensor(job_embeddings)

# Check the shape of the loaded embeddings tensor
print("Shape of loaded job embeddings:", job_embeddings.shape)

df_2 = pd.read_csv("Cleaned_data+scrap (1).csv")
print(df_2.shape)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# Load the pre-trained model
model = AutoModel.from_pretrained("MohammedDhiyaEddine/job-skill-sentence-transformer-tsdae").to(device)
tokenizer = AutoTokenizer.from_pretrained("MohammedDhiyaEddine/job-skill-sentence-transformer-tsdae")

# Define the route to handle the POST request from the frontend
def clean_text(text):
    # Remove numbers, emails, and links
    cleaned_text = re.sub(r'\b[0-9]+\b', '', text)  # Remove numbers
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)  # Remove emails
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)  # Remove links
    # Remove spaces and special characters using regexa
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove multiple spaces
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', cleaned_text)  # Remove non-alphanumeric characters except spaces
    return cleaned_text


# Generate user embedding
def generate_user_embedding(cleaned_text):
    encoded_input = tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state.mean(dim=1)  # Mean pooling to get sentence embedding
    return embeddings


# Generate cosine similarity matrix and merged DataFrame
def get_cosine_matrix(user_embedding):
    tensor1 = user_embedding
    tensor2 = job_embeddings

    permutations = itertools.product(tensor1, tensor2)

    Embedding_df = pd.DataFrame(permutations, columns=['tensor1', 'tensor2'])
    Embedding_df['Index'] = range(tensor2.shape[0])
    Embedding_df.set_index('Index', inplace=True)
    Embedding_df['cosine_similarity'] = np.zeros((Embedding_df.shape[0]))

    for i in range(math.ceil(Embedding_df.shape[0])):
        tensor1 = Embedding_df.iloc[i]['tensor1']
        tensor2 = Embedding_df.iloc[i]['tensor2']
        cosine_similarity_value = torch.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()
        Embedding_df.at[i, 'cosine_similarity'] = cosine_similarity_value

    Embedding_df = Embedding_df.sort_values(by='cosine_similarity', ascending=False)
    print(Embedding_df.shape)
    print("df..\n",Embedding_df)
    merged_df = pd.merge(Embedding_df, df_2, on='Index').set_index('Index')

    return Embedding_df, merged_df


@app.route('/api/process-resume', methods=['POST'])
def process_resume():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Extract text from the resume PDF using PyMuPDF
        pdf_reader = PyPDF2.PdfReader(resume_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text().lower()
        print('Text->',text)
        # resume_content = resume_file.read()
        text = clean_text(text)
        annotations = skill_extractor.annotate(text)
        # Initialize an empty set to store unique extracted skills
        unique_extracted_skills = set()

        # Extract skills from full_matches
        for match in annotations['results']['full_matches']:
            unique_extracted_skills.add(match['doc_node_value'])

        # Extract skills from ngram_scored
        for scored_match in annotations['results']['ngram_scored']:
            unique_extracted_skills.add(scored_match['doc_node_value'])

        # Convert the set back to a list if needed
        unique_extracted_skills_list = list(unique_extracted_skills)

        # Join all unique extracted skills with commas
        comma_separated_unique_skills = ' '.join(unique_extracted_skills_list)

        # Print the comma-separated unique skills
        print("Extractor....\n",comma_separated_unique_skills)
        user_embedding = generate_user_embedding(comma_separated_unique_skills)

        # Prin the generated embeddings (for testing purposes)
        # print(user_embedding)
        CosineMatrix, merged_df = get_cosine_matrix(user_embedding)
        print(merged_df[['tensor1', 'tensor2', 'Position', 'cosine_similarity']])
        # Get top 10 job recommendations from the merged DataFrame
        top_10_recommendations = merged_df['Position'].head(10).tolist()
        print(top_10_recommendations)
        return jsonify(message="Sentence embeddings generated successfully", jobRecommendations=top_10_recommendations)
    else:
        return jsonify(error="Invalid request")
#/////////////////////////////////////////////////////////////////////////////////////
from io import BytesIO

# # Read skills from CSV file
# file_path = 'Book2.csv'
# with open(file_path, 'r') as file:
#     csv_reader = csv.reader(file)
#     skills = [row[0] for row in csv_reader]
#
# # Create pattern dictionaries from skills
# skill_patterns = [[{'LOWER': skill}] for skill in skills]
#
# # Create a Matcher object
# matcher = Matcher(nlp.vocab)
#
# # Add skill patterns to the matcher
# for pattern in skill_patterns:
#     matcher.add('Skills', [pattern])
#
# # Function to extract skills from text
#
# def extract_text_from_pdf(pdf_content):
#     doc = nlp(pdf_content)
#     matches = matcher(doc)
#     extracted_skills = set()
#     for match_id, start, end in matches:
#         skill = doc[start:end].text
#         extracted_skills.add(skill)
#     print(extracted_skills)
#     return extracted_skills


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
