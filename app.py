import streamlit as st
import requests
import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login, HfApi
import os
import subprocess
# Configuration for API
API_KEY = st.secrets["openai"]["api_key"]
ENDPOINT = "https://rayee-m3lv0e7b-westeurope.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"
HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Hardcoded file path and log file
FILE_PATH = "Synopsis_text.docx"  # Replace with the correct document path
LOG_FILE = "user_queries_log.txt"  # File to log unanswered queries

api_token = st.secrets["huggingface"]["api_token"]
if api_token is None:
    st.error("Hugging Face API Key not found. Please set it in the Secrets Manager.")
else:
    login(token=api_token)

# Function to load and chunk the document
def load_and_chunk_document(file_path, chunk_size=500):
    try:
        doc = Document(file_path)
        full_text = "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        return chunks
    except Exception as e:
        return f"Error reading document: {e}"

# Find the most relevant chunk based on the question
def find_relevant_chunk(question, chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question] + chunks)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    most_relevant_index = similarities.argmax()
    return chunks[most_relevant_index], similarities[most_relevant_index]

# Query the AI model
def query_ai_model(question, relevant_chunk):
    payload = {
        "messages": [
            {"role": "system", "content": "You are an AI assistant answering questions about a project synopsis."},
            {"role": "user", "content": f"Based on this project synopsis: {relevant_chunk}, answer the question: {question}"},
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800,
    }
    try:
        response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
    except requests.RequestException as e:
        return f"Error while querying the AI: {':'.join(str(e).split(':')[:2])}"
    return answer




# Function to log and commit to Git
def log_and_commit_to_git(email, query, log_file="user_queries_log.txt"):
    try:
        # Log the query into the file
        with open(log_file, "a") as file:
            file.write(f"Email: {email}, Query: {query}\n")
        
        # Configure Git
        os.system('git config --global user.email "your-email@example.com"')
        os.system('git config --global user.name "your-username"')

        # Use the GitHub token for authentication
        github_token = os.getenv("github_token")
        #github_token = st.secrets["github"]["token"]
        if not github_token:
            raise ValueError("GitHub token not found in secrets.")

        repo_url = f"https://{github_token}@github.com/rayzcell/document_search_genai.git"
        subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True)

        # Stage, commit, and push
        subprocess.run(["git", "add", log_file], check=True)
        subprocess.run(["git", "commit", "-m", f"Logged query from {email}"], check=True)
        subprocess.run(["git", "push"], check=True)

        print("Query logged and committed to Git successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Log unanswered queries
def log_unanswered_query(email, question):
    try:
        log_and_commit_to_git(email, question)
        print("Query logged and committed to Git.")
    except Exception as e:
        print(f"Error while logging the query: {e}")

# Streamlit app layout
st.title("Project Synopsis QA System")
st.write(
    "Ask questions about the project. If no answer is found, you can provide your email for follow-up, and your query will be logged."
)

question = st.text_input("Enter your Question")
email = st.text_input("Enter your Email (Optional)")

if st.button("Submit"):
    # Load and chunk the document
    chunks = load_and_chunk_document(FILE_PATH)
    if isinstance(chunks, str):  # If error occurs
        st.error(chunks)
    else:
        # Find the most relevant chunk
        relevant_chunk, similarity = find_relevant_chunk(question, chunks)
        if similarity < 0.2:
            if email:
                log_unanswered_query(email, question)
                st.warning(f"We couldn't find relevant information. Your query has been logged. We'll email you at {email} with a detailed response.")
            else:
                st.warning("We couldn't find relevant information. Please provide your email for follow-up.")
        else:
            # Query the AI model
            answer = query_ai_model(question, relevant_chunk)
            st.success(answer)
