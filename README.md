# Document Management REST API

This Document Management REST API provides functionality for uploading, indexing, and searching within text documents and PDFs. It supports uploading documents in text or PDF format, submitting text for indexing, and searching for specific words with options to highlight occurrences in the documents.

## Features

- **Document Upload:** Support for uploading text and PDF documents.
- **Text Submission for Indexing:** Allows submission of text directly for indexing.
- **Search and Highlight:** Ability to search for words within documents and highlight occurrences.

## Getting Started

## Installation

To use this API, first clone the repository to your local machine:

```bash
git clone https://github.com/your_username/your_repository.git
```
Navigate to the cloned repository and then app:
```bash
cd your_repository
cd app
```
Next, install the required dependencies:
```bash
pip install -r requirements.txt
```
## Running the API

Run the API server using 
```bash
python main.py
```




### Prerequisites

Before running the API, ensure you have the following dependencies installed:

- FastAPI
- Uvicorn
- PyMuPDF (fitz)
- Transformers
- LLaMA Index and its dependencies
- ChromaDB for document storage and indexing

### Initialization

Initialize the components required for embedding, storage, and text splitting:

```python
embed_model_tokenizer, embed_model, service_context, text_splitter, chroma_collection, storage_context, vector_store = initialize_components()
```

## Running the API

