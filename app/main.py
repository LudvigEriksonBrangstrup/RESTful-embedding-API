import json
import os
import re
import tempfile
import uuid
import shutil
import torch

from fastapi import FastAPI, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from transformers import AutoTokenizer

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.storage.storage_context import StorageContext
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter



def initialize_components(chunk_size=120, chunk_overlap=20):
    """
    Initializes the components needed for the application.

    Parameters:
    chunk_size (int): The size of the chunks that the text should be split into. Default is 120.
    chunk_overlap (int): The number of characters that should overlap between chunks. Default is 20.

    Returns:
    tuple: A tuple containing the initialized components: 
    - service_context (ServiceContext): The service context.
    - text_splitter (RecursiveCharacterTextSplitter): The text splitter.
    - chroma_collection (Collection): The database collection.
    - storage_context (StorageContext): The storage context.
    - vector_store (ChromaVectorStore): The vector store.
    """
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    embed_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    db2 = chromadb.PersistentClient(path="database/chroma_db")
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
     
    return service_context, text_splitter, chroma_collection, storage_context, vector_store



def delete_all_data(input_chroma_collection):
    """
    Deletes all data from the given Chroma collection and the document folders.

    This function retrieves all IDs from the given Chroma collection and deletes the corresponding documents.
    It also deletes all files and subdirectories in the 'highlighted_documents' and 'documents' directories.

    Parameters:
    input_chroma_collection (Collection): The Chroma collection to delete data from.
    """
    all_ids = input_chroma_collection.peek()['ids']
    input_chroma_collection.delete(all_ids)

    directories = ['highlighted_documents', 'documents']
    for directory in directories:
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print(f'Failed to delete {directory}. Reason: {e}')
    return

def initialize_agent_components():
    """
    Initializes the components needed for the agent.

    Returns:
    tuple: A tuple containing the initialized components: 
    - model_sw3 (transformers.PreTrainedModel): The language model.
    - tokenizer_sw3 (transformers.PreTrainedTokenizer): The tokenizer.
    - device (torch.device): The device to run the model on.
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    # Initialize Variables
    model_name_sw3 = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Tokenizer & Model
    tokenizer_sw3 = AutoTokenizer.from_pretrained(model_name_sw3)
    model_sw3 = AutoModelForCausalLM.from_pretrained(model_name_sw3)
    model_sw3.eval()
    model_sw3.to(device)

    return model_sw3, tokenizer_sw3, device

def improve_query_with_agent(initial_text_results, query_str, model_sw3, tokenizer_sw3, device): 
    """ 
    NOTE this is a experimental function and have not been implemented throughly.
    
    Improves a query based on initial text results using a language model.
    This function takes a list of initial text results and a query string as input.
    It generates a prompt based on the text results and the query, and feeds the prompt to a language model.
    The model generates a response, which is then cleaned and returned as the improved query.

    Parameters:
    initial_text_results (list): The initial text results to base the prompt on.
    query_str (str): The original query string.
    model_sw3 (transformers.PreTrainedModel): The language model to use for generating the response.
    tokenizer_sw3 (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the prompt and decoding the response.
    device (torch.device): The device to run the model on.

    Returns:
    str: The improved query.
    """
   
    document_context = ""
    for result in initial_text_results:
        document_context += result.text
        
    prompt = f"""
    <|endoftext|><s>
    User:
    {document_context}

    {query_str}

    Vad frågas det om? 
    <s>
    Bot:
    """

    prompt = prompt.strip()

    input_ids = tokenizer_sw3(prompt, return_tensors="pt")["input_ids"].to(device)

    generated_token_ids = model_sw3.generate(
        inputs=input_ids,
        max_new_tokens=80,
    )[0]

    generated_text = tokenizer_sw3.decode(generated_token_ids)

    bot_output = re.search(r'Bot:([\s\S]*)', generated_text)

    if bot_output:
        cleaned_bot_output = bot_output.group(1).strip()
        improved_query = cleaned_bot_output.rsplit('<s><|endoftext|>', 1)[0].strip()
        improved_query = improved_query.rsplit('?', 1)[0].strip()
        print(improved_query)
    else:
        print("Bot output not found.")
        improved_query = query_str
    return improved_query


def get_unique_document_names(input_chroma_collection):
    all_ids = input_chroma_collection.get()

    unique_names = set()  # Use a set to ensure uniqueness

    for item in all_ids['metadatas']: 
        # TODO find a simpler way to get the file name
        node_content = item['_node_content']
        content = json.loads(node_content)
        file_name = content.get('metadata').get('file_name')
        unique_names.add(file_name)

    return list(unique_names)



def index_document(document_path, input_storage_context, input_text_splitter, input_service_context):
    """
    This function reads a document from the specified path, splits it into chunks, 
    generates embeddings for each chunk using the provided embedding model, and stores 
    the embeddings in the provided storage context for later retrieval.

    Parameters:
    document_path (str): The path to the document to index.
    input_storage_context (StorageContext): The storage context to store the embeddings in.
    input_text_splitter (RecursiveCharacterTextSplitter): The text splitter to use to split the document into chunks.
    input_service_context (ServiceContext): The service context that provides the language model and embedding model.
    """
    uploaded_document = SimpleDirectoryReader(input_files=[document_path]).load_data()
    VectorStoreIndex.from_documents(uploaded_document, service_context=input_service_context,storage_context=input_storage_context, show_progress=True, text_splitter = input_text_splitter )
    return



def retrieve_results(input_index, num_results, query_str, input_filename ,num_queries=1):
    """
    Retrieves the top matching results for a given query from a specified file.

    This function creates a retriever using the provided index, sets the number of top results to return, 
    and the number of extra queries to generate. It applies a filter to only retrieve results from the specified file.
    It then retrieves the top matching results for the provided query string.

    Parameters:
    input_index (Index): The index to retrieve results from.
    num_results (int): The number of top results to return.
    query_str (str): The query string to retrieve results for.
    input_filename (str): The name of the file to retrieve results from.
    num_queries (int, optional): The number of extra queries to generate. Default is 1.

    Returns:
    list: A list of the top matching results.
    """

    filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=input_filename)])

    retriever = input_index.as_retriever(verbose=True, similarity_top_k=num_results, num_queries = num_queries, filters=filters) # num_queries is the number of extra generated queries, 1 disables extra queries
    results = retriever.retrieve(query_str)
    return results



def get_combined_indexes(results):
    """
    This function takes a list of results, each containing a start index, end index, and page number (Optional).
    It combines overlapping indexes into a single index, and returns a list of these combined indexes.

    Parameters:
    results (list): The results of a document retrieval operation. Each result should have a 'node' attribute 
    with 'start_char_idx', 'end_char_idx', and 'metadata' containing 'page_label'. If the metadata does not
    contain 'page_label' due to the document not having pages, 'page_label' is set to 1.

    Returns:
    list: A list of tuples, each containing a start index, end index, and page number. Overlapping indexes 
    are combined into a single index.
    """

    def combine_overlapping_indexes(indexes):
        indexes.sort(key=lambda x: (x[2], x[0]))  # Sort by page number, then by start index
        combined_indexes = []

        for start, end, page in indexes:
            if combined_indexes and start <= (combined_indexes[-1][1] - 10) and page == combined_indexes[-1][2]:
                combined_indexes[-1] = (combined_indexes[-1][0], max(end, combined_indexes[-1][1]), page)
            else:
                combined_indexes.append((start, end, page))
        return combined_indexes
    
    indexes = []  # To store tuples of (start_idx, end_idx, page_number)

    # Store start and end indexes
    for result in results:
        print("tesxt: ", result.text)
        start_idx = result.node.start_char_idx
        end_idx = result.node.end_char_idx
        try:
            page_number = int(result.node.metadata['page_label'])
        except KeyError:
            page_number = 1
        indexes.append((start_idx, end_idx, page_number))

    combined_indexes = combine_overlapping_indexes(indexes)
    
    return combined_indexes



def create_text_file(input_text, dirpath, filename=None):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # If filename is not provided, generate a unique one
    if filename is None:
        existing_files = os.listdir(dirpath)
        counter = 1
        filename = f"document({counter}).txt"
        while filename in existing_files:
            counter += 1
            filename = f"document({counter}).txt"

    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'w') as f:
        f.write(input_text)
    return os.path.abspath(filepath), filename



def process_input_document(document, input_filename = None):
    #dirpath="documents"
    dirpath = "PDFs_test"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # If the input is a string
    if isinstance(document, str):
        return create_text_file(document, dirpath, input_filename)
    
    filename_from_document = secure_filename(input_filename.__str__())
    file_path = os.path.join(dirpath, filename_from_document) 

    with open(file_path, "wb+") as file_object:
        file_object.write(document.read())

    return file_path, filename_from_document
 


def highlight_text_in_html(filepath, combined_indexes, background_color="rgb(255, 204, 51)"):
    """
    Highlights specified text in an TXT document.

    This function takes a path to a TXT file and a list of indexes as input.
    Each index in the list is a tuple containing the start index, end index, and page number of the text to be highlighted.
    The function opens the HTML document, reads its content, and then iterates over the indexes.
    For each index, it extracts the corresponding text, wraps it in a span element with a specified background color and bold font weight, and adds it to the new HTML content.
    It also adds any non-highlighted text before and after each highlighted section.
    Finally, it wraps the new HTML content in a basic HTML structure, saves it in a new HTML document, and returns the path to the new document.

    Parameters:
    filepath (str): The path to the HTML document.
    combined_indexes (list): A list of tuples, where each tuple contains the start index, end index, and page number of the text to be highlighted.
    background_color (str, optional): The background color to use for highlighting. Defaults to "rgb(255, 204, 51)".

    Returns:
    str: The path to the new HTML document with the highlighted text.
    """
    output_pdf_path = "highlighted_documents"
    if not os.path.exists(output_pdf_path):
        os.makedirs(output_pdf_path)
    
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    html_content = ""
    previous_end_idx = 0

    for start_idx, end_idx, page_number in combined_indexes:
        # Add non heiglighted text before the current highlighted section (if any)
        html_content += content[previous_end_idx:start_idx]
        
        style = f"background-color: {background_color}; font-weight: bold;"
        highlighted_section = f'<span style="{style}">{content[start_idx:end_idx]}</span>'
        html_content += highlighted_section
        
        previous_end_idx = end_idx

    # Add any remaining non heiglighted text
    html_content += content[previous_end_idx:]
    html_content = html_content.replace('\n', '<br>')

    # Wraper
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Document</title>
    </head>
    <body>{html_content}</body>
    </html>"""

    output_HTML_text_path = 'highlighted_documents/highlighted_text.html'

    with open(output_HTML_text_path, 'w') as f:
            f.write(html_content)

    return output_HTML_text_path



def highlight_text_in_pdf(pdf_path, indexes):
    """
    Highlights specified text in a PDF document.

    This function takes a path to a PDF document and a list of indexes as input.
    Each index in the list is a tuple containing the start index, end index, and page number of the text to be highlighted.
    The function opens the PDF document, creates a new directory for the output if it doesn't exist, and then iterates over the indexes.
    For each index, it loads the corresponding page, extracts the text, finds the target text, and then searches for the target text in the page to get its bounding box.
    It then highlights each instance of the target text found in the page.
    Finally, it saves the modified PDF document in the output directory and returns the path to the modified document.

    Parameters:
    pdf_path (str): The path to the PDF document.
    indexes (list): A list of tuples, where each tuple contains the start index, end index, and page number of the text to be highlighted.

    Returns:
    str: The path to the modified PDF document.
    """


    doc = fitz.open(pdf_path)

    output_pdf_path = "highlighted_documents"
    if not os.path.exists(output_pdf_path):
        os.makedirs(output_pdf_path)
    
    filename = os.path.basename(pdf_path)
    filepath = os.path.join(output_pdf_path, filename)
    
    for start_idx, end_idx, page_number in indexes:
       
        page = doc.load_page(page_number - 1)  
        
        # Extract text from the page to find the exact position of your text
        text = page.get_text("text")
        target_text = text[start_idx:end_idx]
        
        # Search for the target text in the page to get its bbox (bounding box)
        text_instances = page.search_for(target_text)
        
        # Highlight each instance found
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
    
    doc.save(filepath, garbage=4, deflate=True)
    doc.close()
    return filepath



# =============================================================================


service_context, text_splitter, chroma_collection, storage_context, vector_store = initialize_components()

document_list = get_unique_document_names(chroma_collection)

use_agent = False

if use_agent:
    model_sw3, tokenizer_sw3, device = initialize_agent_components()


print("================================================================================================")
print("initializing components")
print("================================================================================================")

app = FastAPI()

# Enabeling acess to the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


  


@app.get("/documents/")
def get_documents():
    """
    This function is a FastAPI endpoint that handles GET requests to the "/documents/" URL.
    When called, it returns a JSON object containing a list of all documents.

    Returns:
    dict: A dictionary with a single key-value pair. The key is "documents", and the value is a list of all documents.
    """
    return {"documents": document_list}


@app.post("/index_all/")
async def index_all_documents():
    """
    Indexes all documents in the 'PDFs_test' directory.

    This function is a FastAPI endpoint that handles POST requests to the "/index_all/" URL.
    It iterates over all the files in the 'PDFs_test' directory and calls `index_document()` on each one.

    Returns:
    dict: A dictionary with a status message.
    """
    directory = 'PDFs_test'
    count = 0
    size = len(os.listdir(directory))
    for filename in os.listdir(directory):
        print("indexing document number ", count, " of ", size)
        count += 1
        document_path = os.path.join(directory, filename)
        index_document(document_path, storage_context, text_splitter, service_context)

    return {"status": "All documents indexed successfully"}

@app.post("/file_upload/")
async def create_upload_file(file: UploadFile):
    """
    Uploads a file and indexes it for later retrieval.

    This function is a FastAPI endpoint that handles POST requests to the "/file_upload/" URL.
    It takes an uploaded file as input, checks if the file is provided and if it's a PDF or TXT file.
    If the checks pass, it processes the document, adds the filename to the document list, 
    and indexes the document for later retrieval.

    Parameters:
    file (UploadFile): The file to upload and index.

    Returns:
    dict: A dictionary with a status message and the filename of the uploaded file.
    """
    if file:
        document = file.file
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type == "text/plain" and not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid document type. Please provide a PDF file, a TXT file")

    document_path, filename = process_input_document(document, filename)
    document_list.append(filename)
    
    index_document(document_path, storage_context, text_splitter, service_context)
    return {"status": "Text uploaded successfully", "filename": filename}


class TextUpload(BaseModel):
    text: str

@app.post("/text_upload/")
async def upload_text(payload: TextUpload):
    """
    Uploads a text string and indexes it for later retrieval.

    This function is a FastAPI endpoint that handles POST requests to the "/text_upload/" URL.
    It takes a string of text as input, checks if the text is provided.
    If the check passes, it processes the text, adds the filename to the document list, 
    and indexes the document for later retrieval.

    Parameters:
    text (str): The string of text to upload and index.

    Returns:
    dict: A dictionary with a status message and the filename of the uploaded text.
    """
    print("starting upload")
    if payload.text:
        text = re.sub(r'\r?\n', '<br>', payload.text)  # Replace newlines with <br>
    else:
        raise HTTPException(status_code=400, detail="No text provided")
   
    document_path, filename = process_input_document(text)
    document_list.append(filename)

    index_document(document_path, storage_context, text_splitter, service_context)

    return {"status": "Text uploaded successfully", "filename": filename}

"""
class HighlightRequest(BaseModel):
    filename: str
    search_word: str

@app.post("/highlight/")
def highlight_text(request: HighlightRequest):
    filename = request.filename
    search_word = request.search_word    
"""

@app.get("/highlight/")
def highlight_text(filename: str = Query(..., description="The name of the file to search"), search_word: str = Query(..., description="The word to search for")):    
    """
    Highlights relevant parts based on search words in a document and returns the document.

    This function is a FastAPI endpoint that handles GET requests to the "/highlight/" URL.
    It takes a filename and a search word as input. It retrieves the document with the given filename,
    highlights all occurrences of the search word in the document, and returns the document.

    If the document is a PDF, it returns the document as a PDF with the highlighted text.
    If the document is not a PDF, it returns the document as an HTML file with the highlighted text.

    Parameters:
    filename (str): The name of the file to search.
    search_word (str): The word to search for.

    Returns:
    FileResponse: A FileResponse object containing the document with the highlighted text.
    """    
    if not filename or not search_word:
        raise HTTPException(status_code=400, detail="Both filename and search_word must be provided")
    if filename not in document_list:
        raise HTTPException(status_code=400, detail="Document not found")

    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    #document_path = os.path.join("documents", filename)
    document_path = os.path.join("PDFs_test", filename)

    num_results = 1  # TODO make configurable via query parameters
    results = retrieve_results(index, num_results, search_word, filename)

    if use_agent:
        improved_query = improve_query_with_agent(results, search_word, model_sw3, tokenizer_sw3, device)
        results = retrieve_results(index, num_results, improved_query, filename)
        
    combined_indexes = get_combined_indexes(results)

    #print("combined indexes: ", combined_indexes.__sizeof__())

    if combined_indexes.__sizeof__() == 3:
        raise HTTPException(status_code=400, detail="bad searchword")
    
    if filename.lower().endswith('.pdf'):
        filepath = highlight_text_in_pdf(document_path, combined_indexes)
        return FileResponse(filepath, media_type='application/pdf', filename=os.path.basename(filepath))
    else:
        filepath = highlight_text_in_html(document_path, combined_indexes)
        return FileResponse(filepath, media_type='text/html')



@app.delete("/clear_database/")
async def clear_database():
    """
    Clears all data from the database and document folders.

    This function is a FastAPI endpoint that handles DELETE requests to the "/clear_database/" URL.
    When called, it deletes all data from the database, document folders and clears the 'document_list'.
    It then returns a success message.

    Returns:
    dict: A dictionary with a single key-value pair. The key is "message", and the value is a success message.
    """
    delete_all_data(chroma_collection)
    document_list.clear()
    return {"message": "Database cleared successfully"}
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)