# Document Embedding & Query Processing with SBERT

This repository provides a pipeline for processing and clustering documents, generating embeddings, and querying them using SBERT (Sentence-BERT). The project leverages multiprocessing to handle large sets of documents, KMeans for clustering, and cosine similarity for ranking documents based on query relevance. The system is designed to scale and utilize GPU resources for faster computations.

## Features
- **Text Preprocessing**: Clean and tokenize text documents.
- **Document Embedding**: Use SBERT to generate dense vector representations (embeddings) of documents.
- **Multiprocessing**: Speed up document processing by leveraging multiple CPU cores.
- **Clustering**: Perform KMeans clustering to group similar documents.
- **Inverted Index**: Create an inverted index for efficient retrieval of documents by clusters.
- **Query Processing**: Preprocess and embed queries, retrieve top-k clusters, and rank documents based on cosine similarity.
- **GPU Acceleration**: Optionally leverage GPU for faster computation during query processing.

## Installation

### Prerequisites
- Python 3.10+
- `torch` for GPU support (optional, recommended)
- `Sentence-Transformers` for document embedding
- `sklearn` for KMeans clustering
- `nltk` for text tokenization


