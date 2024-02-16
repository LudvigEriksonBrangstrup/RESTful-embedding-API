import json
import os
import re
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
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
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    embed_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    db2 = chromadb.PersistentClient(path="database/chroma_db")
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
     
    return embed_model_tokenizer, embed_model, service_context, text_splitter, chroma_collection, storage_context, vector_store


def get_unique_document_names(input_chroma_collection):
    all_ids = input_chroma_collection.get()

    unique_names = set()  # Use a set to ensure uniqueness

    for item in all_ids['metadatas']: 
        # TODO find a simpler way to get the file namea
        node_content = item['_node_content']
        content = json.loads(node_content)
        file_name = content.get('metadata').get('file_name')
        unique_names.add(file_name)

    return list(unique_names)


def index_document(document_path, embed_model, input_storage_context, input_text_splitter, input_service_context):
    uploaded_document = SimpleDirectoryReader(input_files=[document_path]).load_data()
    VectorStoreIndex.from_documents(uploaded_document, service_context=input_service_context,storage_context=input_storage_context, show_progress=True, text_splitter = input_text_splitter )
    return

def retrieve_results(input_index, num_results, query_str, input_filename ,num_queries=1):

    filters = MetadataFilters(filters=[
        ExactMatchFilter(key="file_name", value=input_filename)
    ])

    retriever = input_index.as_retriever(verbose=True, similarity_top_k=num_results, num_queries = num_queries, filters=filters) # num_queries is the number of extra generated queries, 1 disables extra queries
    results = retriever.retrieve(query_str)
    print("size of results: ", len(results))
    return results


def get_combined_indexes(results):

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
    dirpath="documents"
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

    output_pdf_path = 'highlighted_documents/highlighted_text.html'

    with open(output_pdf_path, 'w') as f:
            f.write(html_content)

    return output_pdf_path


def highlight_text_in_pdf(pdf_path, indexes):
    doc = fitz.open(pdf_path)

    output_pdf_path = "highlighted_documents"
    if not os.path.exists(output_pdf_path):
        os.makedirs(output_pdf_path)
    
    filename = os.path.basename(pdf_path)
    filepath = os.path.join(output_pdf_path, filename)
    
    for start_idx, end_idx, page_number in indexes:
        # Access the specific page
        page = doc.load_page(page_number - 1)  # Page numbers are 0-indexed in PyMuPDF
        
        # Extract text from the page to find the exact position of your text
        text = page.get_text("text")
        target_text = text[start_idx:end_idx]
        
        # Search for the target text in the page to get its bbox (bounding box)
        text_instances = page.search_for(target_text)
        
        # Highlight each instance found
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
    
    # Save the modified PDF
    doc.save(filepath, garbage=4, deflate=True)
    doc.close()
    return filepath



# =============================================================================


embed_model_tokenizer, embed_model, service_context, text_splitter, chroma_collection, storage_context, vector_store = initialize_components()

document_list = get_unique_document_names(chroma_collection)


print("================================================================================================")
print("initializing components")
print("================================================================================================")

app = FastAPI()

# New CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#TODO make sure that the API is easy to use without the Front end
  

class TextUpload(BaseModel):
    text: str

class FileUpload(BaseModel):
    file: UploadFile

@app.get("/documents/")
def get_documents():
    return {"documents": document_list}


@app.post("/file_upload/")
async def create_upload_file(file: UploadFile):
    if file:
        document = file.file
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type == "text/plain" and not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid document type. Please provide a PDF file, a TXT file")

    document_path, filename = process_input_document(document, filename)
    document_list.append(filename)
    
    index_document(document_path, embed_model, storage_context, text_splitter, service_context)
    return {"status": "Text uploaded successfully", "filename": filename}


@app.post("/text_upload/")
async def upload_text(payload: TextUpload):
    print("starting upload")
    if payload.text:
        text = re.sub(r'\r?\n', '<br>', payload.text)  # Replace newlines with <br>
    else:
        raise HTTPException(status_code=400, detail="No text provided")
   
    document_path, filename = process_input_document(text)
    document_list.append(filename)

    index_document(document_path, embed_model, storage_context, text_splitter, service_context)

    return {"status": "Text uploaded successfully", "filename": filename}



from fastapi import Query

@app.get("/highlight/")
def highlight_text(filename: str = Query(..., description="The name of the file to search"), search_word: str = Query(..., description="The word to search for")):
    if not filename or not search_word:
        raise HTTPException(status_code=400, detail="Both filename and search_word must be provided")
    if filename not in document_list:
        raise HTTPException(status_code=400, detail="Document not found")

    # The rest of the logic remains the same
    index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
    document_path = os.path.join("documents", filename)

    num_results = 4  # This could be made configurable via query parameters as well
    results = retrieve_results(index, num_results, search_word, filename)



    combined_indexes = get_combined_indexes(results)

    print("combined indexes: ", combined_indexes.__sizeof__())

    if combined_indexes.__sizeof__() == 3:
        raise HTTPException(status_code=400, detail="bad searchword")
    
    
    if filename.lower().endswith('.pdf'):
        filepath = highlight_text_in_pdf(document_path, combined_indexes)
        return FileResponse(filepath, media_type='application/pdf', filename=os.path.basename(filepath))
    else:
        filepath = highlight_text_in_html(document_path, combined_indexes)
        return FileResponse(filepath, media_type='text/html')

    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)