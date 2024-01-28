import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Read job postings CSV into DataFrame
job_postings_df = pd.read_csv("files/job_postings.csv",nrows=2000)
resume_df = pd.read_csv("files/UpdatedResumeDataSet.csv",nrows=2000)

def tokenize_and_filter(text, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))

    # Assuming you have a tokenizer initialized somewhere, replace with your own tokenizer
    tokenizer = RegexpTokenizer(r'\w+')  # Replace with your tokenizer

    tokens = tokenizer.tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    pos_tags = pos_tag(filtered_tokens)
    filtered_words = [word for word, pos in pos_tags if pos in ['NNP', 'NN']]

    if not len(filtered_words) > 1:
        return None
    return ' '.join(filtered_words)

def extract_skills(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

# Apply skill extraction to job titles and descriptions
# job_postings_df['desc'] = job_postings_df['description'].apply(tokenize_and_filter).apply(lambda text: ' '.join(extract_skills(nlp(text))))
job_postings_df['desc'] = job_postings_df['description'].apply(lambda text: ' '.join(extract_skills(nlp(text))))

# Process resumes DataFrame (assuming it has been processed as per the previous code)
resume_df['res'] = resume_df['Resume'].apply(lambda text: ' '.join(extract_skills(nlp(text))))

# Create a CountVectorizer to convert text to vectors
vectorizer = CountVectorizer()

# Transform job posting skills and resume skills into vectors
job_skills_vectors = vectorizer.fit_transform(job_postings_df['desc'])
resume_vectors = vectorizer.transform(resume_df['res'])

# Calculate cosine similarity between job posting skills and resumes
similarity_matrix = cosine_similarity(job_skills_vectors, resume_vectors)

# Find the most similar job posting for each resume
best_matching_job_indices = similarity_matrix.argmax(axis=0)

# Add a new column to the resume DataFrame with the best matching job indices
resume_df['Best_Matching_Job_Index'] = best_matching_job_indices

resume_df['Percentage_Match'] = similarity_matrix.max(axis=0) * 100  # Normalize to percentage scale

# Print the results
for index, row in resume_df[resume_df['Percentage_Match'] > 50].iterrows():
    print(
        f"Resume {index + 1} matches best with Job Posting {row['Best_Matching_Job_Index'] + 1} with a {row['Percentage_Match']:.2f}% match.")
