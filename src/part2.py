from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import os
from multiprocessing import Pool, Manager
import numpy as np
import time
import torch
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import heapq
import pandas as pd
from functools import partial
import csv

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SBERT model on the specified device (GPU or CPU)
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

MAX_TOKENS = 512


# Applying preprocessing
def preprocess_text(text):
    """
    Preprocess the text by:
    - Lowercasing
    - Removing punctuation and special characters
    - Tokenizing
    - Joining tokens back into a string
    """
    text = text.lower()  # Lowercase the text
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize text into words
    return " ".join(tokens)  # Join tokens back into a single string


def preprocess_single_file(file_path):
    """
    Preprocess a single file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        preprocessed_text = preprocess_text(text)
        return os.path.basename(file_path), preprocessed_text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return os.path.basename(file_path), None


def preprocess_single_file(file_path):
    """
    Preprocess a single file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        preprocessed_text = preprocess_text(text)
        return os.path.basename(file_path), preprocessed_text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return os.path.basename(file_path), None


def preprocess_documents_multiprocessing(input_directory, output_directory):
    """
    Preprocess documents in `input_directory` and save them in `output_directory`
    using multiprocessing.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_paths = [
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if file.endswith(".txt")
    ]

    start_time = time.time()

    with Pool() as pool:
        results = pool.map(preprocess_single_file, file_paths)

    for filename, preprocessed_text in results:
        if preprocessed_text:
            output_path = os.path.join(output_directory, filename)
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(preprocessed_text)

    elapsed = time.time() - start_time
    print(f"Preprocessed {len(file_paths)} documents in {elapsed:.2f} seconds.")


# Function to preprocess the query text
def preprocess_query(text):
    """
    Preprocess the query by:
    - Lowercasing
    - Removing punctuation and special characters
    - Tokenizing
    """
    text = text.lower()  # Lowercase the text
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize the text into words
    return " ".join(tokens)  # Join tokens back into a single string


# Function to split text into chunks
def split_text_into_chunks(preprocessed_text, max_tokens=MAX_TOKENS):
    tokens = preprocessed_text.split()  # Tokens are preprocessed already
    chunks = [
        " ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)
    ]
    return chunks


# Function to process a single document
def process_document(file_path, max_tokens=MAX_TOKENS, progress_counter=None):

    with open(file_path, "r", encoding="utf-8") as file:
        preprocessed_text = file.read()

    tokens = preprocessed_text.split()
    if len(tokens) > max_tokens:

        chunks = split_text_into_chunks(preprocessed_text, max_tokens)
        embeddings = [model.encode(chunk) for chunk in chunks]
        aggregated_embedding = np.mean(embeddings, axis=0)
    else:
        aggregated_embedding = model.encode(preprocessed_text)

    if progress_counter is not None:
        progress_counter.value += 1
        if progress_counter.value % 10000 == 0:
            print(f"Processed {progress_counter.value} documents...")
    # Return document ID and its corresponding embedding
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    return doc_id, aggregated_embedding


# Multiprocessing for processing documents
def save_document_embeddings(embeddings, file_path):
    """
    Save document embeddings to a pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(embeddings, file)
    print(f"Document embeddings saved to '{file_path}'.")


def process_documents_multiprocessing_old(
    input_directory, max_tokens=MAX_TOKENS, save_path="document_embeddings.pkl"
):
    print(f"Starting processing of documents in directory: {input_directory}")
    file_paths = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".txt")
    ]

    with Pool() as pool:
        process_func = partial(process_document, max_tokens=max_tokens)
        print(f"Processing {len(file_paths)} documents in parallel...")
        results = pool.map(process_func, file_paths)

    # Save embeddings
    document_embeddings = {doc_id: embedding for doc_id, embedding in results}
    save_document_embeddings(document_embeddings, save_path)
    print(f"Finished processing {len(file_paths)} documents.")
    return document_embeddings


def process_documents_multiprocessing(
    input_directory, max_tokens=MAX_TOKENS, save_path="document_embeddings.pkl"
):
    """
    Process documents in parallel and save their embeddings.
    """
    start_time = time.time()  # Start timing the process
    print(f"Starting processing of documents in directory: {input_directory}")
    file_paths = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".txt")
    ]

    total_files = len(file_paths)
    print(f"Total files to process: {total_files}")

    with Manager() as manager:
        # Create a shared counter
        progress_counter = manager.Value("i", 0)

        # Use multiprocessing to process documents
        with Pool() as pool:
            process_func = partial(
                process_document,
                max_tokens=max_tokens,
                progress_counter=progress_counter,
            )
            print(f"Processing {total_files} documents in parallel...")
            results = pool.map(process_func, file_paths)

        # Create a dictionary for document embeddings
        document_embeddings = {doc_id: embedding for doc_id, embedding in results}

        # Save embeddings to a pickle file
        save_document_embeddings(document_embeddings, save_path)

    # Calculate and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Finished processing {total_files} documents in {total_time:.2f} seconds.")

    return document_embeddings


def save_object(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def cluster_and_build_inverted_index(
    embeddings, k, centroids_path="centroids.pkl", index_path="inverted_index.pkl"
):
    """
    Perform clustering on the document embeddings and build an inverted index.
    After clustering, it saves the centroids and the inverted index to disk.
    """
    # Convert embeddings to matrix form
    doc_ids, embedding_matrix = zip(*embeddings.items())
    embedding_matrix = np.vstack(embedding_matrix)

    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding_matrix)

    # Build inverted index
    inverted_index = {i: [] for i in range(k)}
    for doc_id, label in zip(doc_ids, labels):
        inverted_index[label].append(doc_id)

    # Store centroids for retrieval
    centroids = kmeans.cluster_centers_

    # Save centroids and inverted index to disk
    save_object(centroids, centroids_path)
    save_object(inverted_index, index_path)

    print(f"Centroids and inverted index saved to {centroids_path} and {index_path}")

    return inverted_index, centroids


def preprocess_queries(input_csv, output_csv):
    """
    Preprocess the 'Query' column in the queries CSV and save the result.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Preprocess the 'Query' column
    df["Query"] = df["Query"].apply(preprocess_text)

    # Save the preprocessed data to a new CSV file
    df.to_csv(output_csv, index=False)

    print(f"Preprocessed queries saved to '{output_csv}'.")


def generate_query_embeddings(preprocessed_query_csv, query_embeddings_path):
    queries = pd.read_csv(preprocessed_query_csv)
    query_embeddings = {}
    for _, row in queries.iterrows():
        query_number = row["Query number"]
        query_text = row["Query"]
        embedding = model.encode([query_text])[0]  # Generate embedding for the query
        query_embeddings[query_number] = embedding
    save_object(query_embeddings, query_embeddings_path)
    return query_embeddings


# Function to load the inverted index and centroids from disk
def load_object(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


# Function to retrieve top-k relevant clusters for the query
def retrieve_top_k_clusters(query_embedding, centroids, k):

    # Compute cosine similarities between the query and each centroid
    cosine_similarities = cosine_similarity([query_embedding], centroids)

    # Get the indices of the top-k highest cosine similarities
    top_k_indices = heapq.nlargest(
        k, range(len(cosine_similarities[0])), key=lambda i: cosine_similarities[0][i]
    )

    return top_k_indices


# Function to retrieve documents from the inverted index for the top-k clusters
def retrieve_documents_from_clusters(inverted_index, top_k_indices):

    relevant_documents = []

    for cluster_idx in top_k_indices:
        relevant_documents.extend(inverted_index.get(cluster_idx, []))

    return relevant_documents


# Function to rank documents based on cosine similarity
def rank_documents(query_embedding, relevant_documents, document_embeddings):

    # Prepare a list to store the document similarities
    document_similarities = []

    for doc_id in relevant_documents:
        doc_embedding = document_embeddings.get(doc_id)
        if doc_embedding is not None:
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            document_similarities.append((doc_id, similarity))

    # Sort the documents by similarity score in descending order and select the top-10
    ranked_documents = sorted(document_similarities, key=lambda x: x[1], reverse=True)[
        :10
    ]

    return ranked_documents


def compute_similarity_for_query(
    query_number, query_vector, inverted_index, centroids, document_embeddings, k
):

    top_k_clusters = retrieve_top_k_clusters(
        query_vector, centroids, k
    )  # Retrieve top-k clusters
    relevant_documents = retrieve_documents_from_clusters(
        inverted_index, top_k_clusters
    )  # Get relevant documents
    ranked_documents = rank_documents(
        query_vector, relevant_documents, document_embeddings
    )  # Rank documents
    return query_number, ranked_documents


# Function to compute similarity in parallel using multiprocessing
def compute_similarity_parallel(
    query_embeddings, inverted_index, centroids, document_embeddings, k
):
    """
    Compute cosine similarity between query embeddings and cluster centroids, retrieve top-k clusters,
    and rank documents within those clusters based on cosine similarity, using multiprocessing.

    Args:
        query_embeddings (dict): A dictionary of query embeddings (key: query number, value: embedding).
        inverted_index (dict): A dictionary representing the inverted index (key: cluster index, value: list of documents).
        centroids (numpy.ndarray): The centroids of the clusters.
        document_embeddings (dict): A dictionary representing document embeddings (key: document ID, value: embedding).

    Returns:
        dict: A dictionary where keys are query numbers and values are lists of ranked documents (from the top-k clusters).
    """
    # Prepare the partial function for multiprocessing
    compute_func = partial(
        compute_similarity_for_query,
        inverted_index=inverted_index,
        centroids=centroids,
        document_embeddings=document_embeddings,
        k=k,
    )

    # Prepare the data for multiprocessing
    query_data = [
        (query_number, query_vector)
        for query_number, query_vector in query_embeddings.items()
    ]

    total_queries = len(query_data)
    print(f"Starting computation for {total_queries} queries...")

    start_time = time.time()

    # Create a Pool of workers
    with Pool() as pool:
        # Map the query data to the compute_similarity_for_query function in parallel
        results = pool.starmap(compute_func, query_data)

    # Prepare the ranked results dictionary
    ranked_results = {
        query_number: ranked_documents for query_number, ranked_documents in results
    }

    # Compute and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Finished processing {total_queries} queries in {total_time:.2f} seconds.")

    return ranked_results


def save_ranked_results_to_csv(ranked_results, output_csv):
    """
    Save ranked results (query number and document number) to a CSV file.

    Args:
        ranked_results (dict): Dictionary with query numbers as keys and a list of ranked document numbers as values.
        output_csv (str): Path to the CSV file where results will be saved.
    """
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Query number", "doc_number"])  # Write header

        # Iterate over each query's ranked documents and save the result
        for query_number, ranked_documents in ranked_results.items():
            for doc_name, _ in ranked_documents:  # Ignore the similarity score for now
                # Extract document number from the filename (e.g., output_590.txt -> 590)
                doc_number = os.path.splitext(doc_name)[0].split("_")[1]
                writer.writerow(
                    [query_number, doc_number]
                )  # Write query number and document number

    print(f"Ranked results saved to '{output_csv}'.")


import torch
from torch.nn.functional import cosine_similarity


def compute_similarity_gpu(
    query_embeddings, inverted_index, centroids, document_embeddings, k, batch_size=32
):
    """
    Compute cosine similarity between query embeddings and cluster centroids,
    retrieve top-k clusters, and rank documents using GPU acceleration.

    Args:
        query_embeddings (dict): A dictionary of query embeddings (key: query number, value: embedding).
        inverted_index (dict): Inverted index mapping cluster IDs to document IDs.
        centroids (numpy.ndarray): The centroids of the clusters.
        document_embeddings (dict): A dictionary representing document embeddings (key: document ID, value: embedding).
        k (int): Number of top clusters to consider.
        batch_size (int): Number of queries to process in a batch.

    Returns:
        dict: A dictionary where keys are query numbers and values are lists of ranked documents.
    """
    # Move data to GPU
    centroids = torch.tensor(centroids, device="cuda", dtype=torch.float32)
    document_embeddings_gpu = {
        doc_id: torch.tensor(embedding, device="cuda", dtype=torch.float32)
        for doc_id, embedding in document_embeddings.items()
    }

    ranked_results = {}
    query_numbers = list(query_embeddings.keys())
    query_tensors = [
        torch.tensor(query_embeddings[query_number], device="cuda", dtype=torch.float32)
        for query_number in query_numbers
    ]

    start_time = time.time()
    total_queries = len(query_numbers)
    print(f"Starting GPU computation for {total_queries} queries...")

    # Process queries in batches
    for i in range(0, total_queries, batch_size):
        batch_query_numbers = query_numbers[i : i + batch_size]
        batch_query_tensors = query_tensors[i : i + batch_size]

        for query_number, query_vector in zip(batch_query_numbers, batch_query_tensors):
            # Step 1: Retrieve top-k clusters
            cluster_similarities = cosine_similarity(
                query_vector.unsqueeze(0), centroids
            )
            top_k_clusters = torch.topk(cluster_similarities, k).indices.tolist()

            # Step 2: Retrieve relevant documents
            relevant_documents = [
                doc_id
                for cluster_id in top_k_clusters
                for doc_id in inverted_index.get(cluster_id, [])
            ]

            if not relevant_documents:
                ranked_results[query_number] = []
                continue

            # Step 3: Rank documents
            relevant_doc_tensors = torch.stack(
                [document_embeddings_gpu[doc_id] for doc_id in relevant_documents]
            )
            doc_similarities = cosine_similarity(query_vector, relevant_doc_tensors)

            # Get top-10 documents
            top_docs = torch.topk(
                doc_similarities, min(10, len(relevant_documents))
            ).indices.tolist()
            ranked_documents = [
                (relevant_documents[idx], doc_similarities[idx].item())
                for idx in top_docs
            ]

            ranked_results[query_number] = ranked_documents

        print(
            f"Processed batch {i // batch_size + 1}/{(total_queries + batch_size - 1) // batch_size}..."
        )

    total_time = time.time() - start_time
    print(
        f"Finished GPU processing of {total_queries} queries in {total_time:.2f} seconds."
    )
    return ranked_results


if __name__ == "__main__":
    original_documents_directory = "full_docs/"
    processed_documents_directory = "processed_docs/"
    centroids_path = "centroids.pkl"
    index_path = "inverted_index.pkl"
    input_csv = "queries.csv"  # Input queries file
    output_csv = "preprocessed_queries.csv"
    document_embeddings_path = "document_embeddings.pkl"
    query_embeddings_path = "query_embeddings.pkl"

    # Uncomment the lines depending on if documents are preprocessed, generated embeddings, inverted index and clusters

    # preprocess_documents_multiprocessing(original_documents_directory, processed_documents_directory)

    print("done preprocess_documents_multiprocessing")

    # document_embeddings = process_documents_multiprocessing(processed_documents_directory)

    print("done process_documents_multiprocessing")

    # inverted_index, centroids = cluster_and_build_inverted_index(document_embeddings, k=10, centroids_path=centroids_path, index_path=index_path )

    print("done cluster_and_build_inverted_index")

    # preprocess_queries(input_csv, output_csv)

    print("done query preprocessing")
    # query_embeddings = generate_query_embeddings(output_csv, query_embeddings_path)
    query_embeddings = load_object(query_embeddings_path)
    inverted_index = load_object(index_path)
    centroids = load_object(centroids_path)

    document_embeddings = load_object(document_embeddings_path)

    print("loaded all")

    # number of clusters to use
    k = 5

    # Compute similarity and rank documents for each query
    # Uncomment the line depending on system capablities: we used gpu so we left that

    # ranked_results = compute_similarity_parallel( query_embeddings, inverted_index, centroids, document_embeddings, k)

    ranked_results = compute_similarity_gpu(
        query_embeddings,
        inverted_index,
        centroids,
        document_embeddings,
        k,
        batch_size=100,
    )
    results_csv = "ranked_results.csv"
    save_ranked_results_to_csv(ranked_results, results_csv)
