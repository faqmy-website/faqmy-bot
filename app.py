from typing import List
from fastapi import FastAPI, Response
from fastapi import File, UploadFile
from dotenv import load_dotenv
from haystack.schema import Document as HaystackDocument
from core_stuff import get_document_store, get_nice_response, get_random_folder, get_url_content, save_docs_from_folder
from pydantic_stuff import AutogeneratedDocument, Document, Query, URLQuery
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline
import shutil
import logging


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

load_dotenv()
app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/i/{index}/documents", status_code=201)
def save_document(index: str, document: Document):
    # dirty hack to add name (question) to the beginning of document content
    new_doc = HaystackDocument(content=f'{document.name} {document.content}', meta={"name": document.name})
    get_document_store(index).write_documents([new_doc])
    return {"id": new_doc.id}


@app.get("/i/{index}/documents/{document_id}", status_code=200)
def get_document(index: str, document_id: str, response: Response):
    doc_store = get_document_store(index)
    document = doc_store.get_document_by_id(document_id)
    if document is None:
        response.status_code = 404
        return {"error": "Document not found"}
    return doc_store.get_document_by_id(document_id)


@app.get("/i/{index}/documents/{document_id}/delete", status_code=200)
def delete_document(index: str, document_id: str, response: Response):
    doc_store = get_document_store(index)
    doc_store.delete_documents(ids=[document_id])
    return {"status": "document deleted"}


@app.get("/i/{index}/documents/", status_code=200)
def get_all_document(index: str, response: Response):
    documents = get_document_store(index).get_all_documents()
    if documents is None:
        response.status_code = 404
        return {"error": "Documents not found"}
    return documents


@app.get("/i/{index}/documents/search/{query}", status_code=200)
def search_document(index: str, query: str, response: Response):
    logging.debug(f"Searching for {query}")
    documents = get_document_store(index).query(query)
    return documents


@app.post("/i/{index}/documents/ask", status_code=200)
def ask_document(index: str, query: Query, response: Response):
    doc_store = get_document_store(index)

    retriever = BM25Retriever(doc_store)
    light_pipeline = DocumentSearchPipeline(retriever)

    params = {"Retriever": {"top_k": 2}}
    result = light_pipeline.run(query=query.question, params=params)

    # dirty hack to cut name from document content
    text_facts = []
    for item in result['documents']:
        name = item.meta["name"]
        content = item.content
        if content.startswith(name):
            content = content[len(item.meta["name"]):]
        text_facts.append(f'{name} {content}')
    return get_nice_response(query.question, text_facts)


@app.post("/i/{index}/documents/scan", status_code=200, response_model=List[AutogeneratedDocument])
def scan_website(index: str, query: URLQuery, response: Response):
    content = get_url_content(query.url)
    upload_folder = get_random_folder()
    file_path = upload_folder / 'scan.txt'

    if content is None:
        return []

    with open(file_path, 'w') as f:
        f.write(content)

    return save_docs_from_folder(upload_folder, index)


@app.post("/i/{index}/upload", response_model=List[AutogeneratedDocument])
def upload(index: str, file: UploadFile = File(...)):
    upload_folder = get_random_folder()

    try:
        file_path = upload_folder / file.filename
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return save_docs_from_folder(upload_folder, index)
