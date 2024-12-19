import os
import numpy as np
import re
import string
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
# Load the model
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
MAX_TOKENS = 512
#stop_words = set(stopwords.words("english"))
#punctuation = set(string.punctuation)
#punctuation.update(["``", "''"])
# Folder paths
original_documents_directory = 'full_docs_small_search/'
processed_documents_directory = 'processed_docs/'

#inputs for queries
input_csv = 'dev_small_queries - dev_small_queries.csv'  # Path to your input queries.csv file
output_csv = 'preprocessed_queries.csv'  # Path to save the preprocessed CSV


def preprocess_text(text):
    """
    Preprocess the text by:
    - Lowercasing
    - Removing punctuation and special characters
    - Tokenizing
    - Removing stop words and punctuation
    - Joining tokens back into a string
    """
    text = text.lower()  # Lowercase the text
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize text into words
    #tokens = [word for word in tokens if word not in stop_words and word not in punctuation]  # Remove stop words and punctuation
    return " ".join(tokens)  # Join tokens back into a single string

def preprocess_documents(input_directory, output_directory):
    """
    Preprocess documents in `input_directory` and save them in `output_directory`.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_directory, filename)

            # Read and preprocess the text
            with open(input_path, 'r') as file:
                text = file.read()
            preprocessed_text = preprocess_text(text)

            # Save the preprocessed text
            output_path = os.path.join(output_directory, filename)
            with open(output_path, 'w') as file:
                file.write(preprocessed_text)

    print(f"Preprocessed documents saved in '{output_directory}'.")

### Step 2: Split and Process Preprocessed Documents ###
def split_text_into_chunks(preprocessed_text, max_tokens=MAX_TOKENS):
    """
    Split preprocessed text into chunks based on `max_tokens`.
    """
    tokens = preprocessed_text.split()  # Split by whitespace (tokens are preprocessed already)
    chunks = [" ".join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def save_chunks(chunks, base_filename):
    """
    Save chunks of text as separate files.
    """
    filenames = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_filename}_{i+1}.txt"
        chunk_path = os.path.join(processed_documents_directory, chunk_filename)
        with open(chunk_path, 'w') as file:
            file.write(chunk)
        filenames.append(chunk_path)
    return filenames

def process_preprocessed_documents(input_directory):
    """
    Process preprocessed documents:
    - Check token count
    - Split into chunks if necessary
    - Calculate embeddings
    """
    document_embeddings = {}

    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_directory, filename)

            # Read the preprocessed text
            with open(file_path, 'r') as file:
                preprocessed_text = file.read()

            tokens = preprocessed_text.split()
            print(f"Processing {filename} - Token count: {len(tokens)}")

            if len(tokens) > MAX_TOKENS:
                # Split into chunks
                chunks = split_text_into_chunks(preprocessed_text)
                chunk_files = save_chunks(chunks, os.path.splitext(filename)[0])

                # Generate embeddings for each chunk and aggregate
                embeddings = []
                for chunk_file in chunk_files:
                    with open(chunk_file, 'r') as file:
                        chunk_text = file.read()
                    embedding = model.encode([chunk_text])[0]
                    embeddings.append(embedding)

                # Average embeddings for the document
                aggregated_embedding = np.mean(embeddings, axis=0)
                document_embeddings[os.path.splitext(filename)[0]] = aggregated_embedding

                # Remove original file if it was split
                os.remove(file_path)
                print(f"Document {filename} split into {len(chunks)} chunks and aggregated.")
            else:
                # Generate embedding directly
                embedding = model.encode([preprocessed_text])[0]
                document_embeddings[os.path.splitext(filename)[0]] = embedding
                print(f"Document {filename} processed without splitting.")

    print("All embeddings calculated and linked to original documents.")
    return document_embeddings

def preprocess_queries(input_csv, output_csv):
    """
    Preprocess the 'Query' column in the queries CSV and save the result.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Preprocess the 'Query' column
    df['Query'] = df['Query'].apply(preprocess_text)

    # Save the preprocessed data to a new CSV file
    df.to_csv(output_csv, index=False)

    print(f"Preprocessed queries saved to '{output_csv}'.")

def generate_query_embeddings(preprocessed_query_csv):
    queries = pd.read_csv(preprocessed_query_csv)
    query_embeddings = {}
    for _, row in queries.iterrows():
        query_number = row['Query number']
        query_text = row['Query']
        embedding = model.encode([query_text])[0]  # Generate embedding for the query
        query_embeddings[query_number] = embedding

    return query_embeddings


def compute_similarity(query_embeddings, document_embeddings):
    """
    Compute cosine similarity between query embeddings and document embeddings.

    Args:
        query_embeddings (dict): A dictionary of query embeddings (key: query number, value: embedding).
        document_embeddings (dict): A dictionary of document embeddings (key: document name, value: embedding).

    Returns:
        dict: A dictionary where keys are query numbers and values are lists of
              tuples (document_name, similarity_score) sorted in descending order.
    """
    # Prepare a dictionary to store results
    similarity_results = {}

    # Convert document embeddings to a list of keys and a numpy array of values
    doc_names = list(document_embeddings.keys())
    doc_vectors = np.array(list(document_embeddings.values()))

    # Compute similarity for each query
    for query_number, query_vector in query_embeddings.items():
        # Compute cosine similarity between the query and all documents
        similarities = cosine_similarity([query_vector], doc_vectors)[0]
        
        # Pair document names with their similarity scores
        ranked_docs = sorted(
            zip(doc_names, similarities),
            key=lambda x: x[1],
            reverse=True
        )  # Sort by similarity score in descending order

        # Filter duplicates (keep only unique document names)
        seen_docs = set()
        unique_ranked_docs = []
        for doc_name, score in ranked_docs:
            if doc_name not in seen_docs:
                unique_ranked_docs.append((doc_name, score))
                seen_docs.add(doc_name)

        # Store the top 10 results for the current query
        similarity_results[query_number] = unique_ranked_docs[:10]

    return similarity_results

def save_similarities_to_csv(similarity_results, output_csv):
    """
    Save similarity results to a CSV file.
    
    Args:
        similarity_results (dict): Dictionary where keys are Query numbers, 
                                   and values are lists of tuples (document_name, similarity_score).
        output_csv (str): Path to the output CSV file.
    """
    rows = []

    for query_number, doc_similarities in similarity_results.items():
        for doc_name, similarity in doc_similarities:
            # Extract the document number (integer part) from the document name
            doc_number = int(re.search(r'\d+', doc_name).group())  # Extracts the first number
            rows.append({"Query_number": query_number, "Document_number": doc_number})
    
    # Convert rows into a DataFrame
    df = pd.DataFrame(rows, columns=["Query_number", "Document_number"])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Similarity results saved to '{output_csv}'.")

# Load ground truth and evaluation results
def load_data(ground_truth_file, evaluation_file):
    ground_truth = pd.read_csv(ground_truth_file, names=['query_number', 'doc_number'])
    evaluation = pd.read_csv(evaluation_file, names=['query_number', 'doc_number'])
    return ground_truth, evaluation

# Compute Precision@k and Recall@k for a single query
def precision_recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = set(retrieved_k).intersection(set(relevant_docs))
    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
    return precision, recall

# Evaluate mean Precision@k and Recall@k
def evaluate_metrics(ground_truth, evaluation, k_values):
    # Group ground truth by query
    relevant_docs_dict = ground_truth.groupby('query_number')['doc_number'].apply(list).to_dict()

    # Group evaluation results by query
    retrieved_docs_dict = evaluation.groupby('query_number')['doc_number'].apply(list).to_dict()

    # Initialize metrics
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}

    # Iterate through each query in evaluation results
    for query_number, retrieved_docs in retrieved_docs_dict.items():
        relevant_docs = relevant_docs_dict.get(query_number, [])

        for k in k_values:
            precision, recall = precision_recall_at_k(retrieved_docs, relevant_docs, k)
            precision_at_k[k].append(precision)
            recall_at_k[k].append(recall)

    # Compute mean metrics
    mean_precision_at_k = {k: sum(precision_at_k[k]) / len(precision_at_k[k]) for k in k_values}
    mean_recall_at_k = {k: sum(recall_at_k[k]) / len(recall_at_k[k]) for k in k_values}

    return mean_precision_at_k, mean_recall_at_k

# Step 1: Preprocess original documents
#preprocess_documents(original_documents_directory, processed_documents_directory)

# Step 2: Process preprocessed documents and generate embeddings
#document_embeddings = process_preprocessed_documents(processed_documents_directory)
#print(document_embeddings["output_1"])
#Step3: Preprocess queries
#preprocess_queries(input_csv, output_csv)
#query_embeddings = generate_query_embeddings('preprocessed_queries.csv')
#similarities = compute_similarity(query_embeddings,document_embeddings)
#output_csv = "similarity_results.csv"
#save_similarities_to_csv(similarities, output_csv)


#Step 3 Metrics
# Example usage
ground_truth_file = "dev_query_results_small.csv"
evaluation_file = "similarity_results.csv"
ground_truth, evaluation = load_data(ground_truth_file, evaluation_file)
k_values = [1, 3, 5, 10]
mean_precision_at_k, mean_recall_at_k = evaluate_metrics(ground_truth, evaluation, k_values)
print("Mean Precision@k:", mean_precision_at_k)
print("Mean Recall@k:", mean_recall_at_k)





