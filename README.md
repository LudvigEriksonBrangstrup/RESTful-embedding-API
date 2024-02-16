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

## Docker Deployment

Alternatively, you can create a Docker container for the API:

```bash
docker build -t myapi .
docker run -d --name myfastapi -p 8080:8080 myapi
```

## API Endpoints

### Upload Document
- **Endpoint:** `/file_upload/`
- **Method:** `POST`
- **Description:** Uploads a document (text or PDF).
- **Request:** Multipart form data with the file. The file field should be named `file`.
- **Curl Example:** 
```bash
curl -X POST -F 'file=@examplefile.pdf' http://localhost:8000/file_upload/
```


### Upload Text
- **Endpoint:** `/text_upload/`
- **Method:** `POST`
- **Description:** Uploads a document as raw text.
- **Request:** JSON payload with the `text` field containing the text you want to upload.
- **Curl Example:** 
```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"your text here"}' http://localhost:8000/text_upload/
```

### Highlight Text
- **Endpoint:** `/highlight/`
- **Method:** `GET`
- **Description:** Uses embeddings to find the best match to the search words in a document and returns the document with the matched words highlighted.
- **Request:** Query parameters `filename` (the name of the file to search) and `search_word` (the words to find the best match for).
- **Curl Example:** 
```bash
curl "http://localhost:8000/highlight/?filename=your_filename&search_word=your_search_word"
```

### Get Documents
- **Endpoint:** `/documents/`
- **Method:** `GET`
- **Description:** Returns a list of all available documents for querying.
- **Curl Example:** 
```bash
curl http://localhost:8000/documents/
```

### Clear Database
- **Endpoint:** `/clear_database/`
- **Method:** `DELETE`
- **Description:** Clears all data from the database and document folders.
- **Curl Example:** 
```bash
curl -X DELETE http://localhost:8000/clear_database/
```

## Error Handling

The API uses standard HTTP responses to indicate the success or failure of an API request:

- `200 OK` - The request was successful.
- `400 Bad Request` - There was a problem with the request.
- `404 Not Found` - The requested resource was not found.
- `500 Internal Server Error` - There was a problem on the server.


