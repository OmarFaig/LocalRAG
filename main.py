import os
from PyPDF2 import PdfReader
import chromadb
from sentence_transformers import   SentenceTransformer
from chromadb.utils import embedding_functions
from llama_cpp import Llama
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
# Connect to the local database
client = chromadb.Client()

def read_file(filepath):
    # Check if the file exists
    if not os.path.isfile(filepath):
        return "Error: File not found"
    
    # Open the PDF file
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        content =""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            content += page.extract_text() + "\n"
        return content
    
def chunk_text(text,chunk_size=200):
    """Split text into smaller chunks for embedding """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_in_chromadb(collection_name, texts):
    """Store the text chunks in the database with embeddings"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name = collection_name)

    for i,text in enumerate(texts):
        embedding = embedding_model.encode(text).tolist()
        collection.add(ids=[str(i)], embeddings=[embedding],metadatas=[{"text":text}])
    print(f"Stored {len(texts)} chunks in the database")

   #document = {
   #    "id": os.path.basename(file_path),
   #    "content": content
   #}
   #collection.upsert(document)

def retrieve_from_chromadb(collection_name, query, top_k = 3):
    """Retrieve documents from the database that match the query"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(collection_name)

    embeddings = embedding_model.encode(query).tolist()
    #print(f"Embeddings : {embeddings}")
    results = collection.query(query_embeddings = [embeddings], n_results = top_k)
    retrieved_texts = [item["text"] for item in results["metadatas"][0]]

    return retrieved_texts

def generate_response_with_llm (retrieved_texts,query):
    """Generate a response using the LLM model"""
    llama = Llama(model_path="llama-2-7b-chat-codeCherryPop.Q3_K_S.gguf")
    context ="\n".join(retrieved_texts)[:400]
    prompt = f"Context : {context}\nQuery : {query}\nResponse :"
    response = llama(prompt,max_tokens=100)
    return response["choices"][0]["text"].strip()

app.mount ("/static",StaticFiles(directory="static"),name="static")
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the file to a temporary directory
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    content = read_file(file_location)
    store_in_chromadb(collection_name="localRAG_pdfs", texts=chunk_text(content))
    return {"info" : f"file '{file.filename}' saved at '{file_location}'"}

@app.post("/query/")
async def query_rag(query: str = Form(...)):
    retrieved_texts = retrieve_from_chromadb(collection_name="localRAG_pdfs", query=query)
    response = generate_response_with_llm(retrieved_texts,query)
    return {"response": response}
@app.get("/",response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(),status_code=200)

def main(pdf_dir):

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            content = read_file(file_path)
            chunks = chunk_text(content)
            #print(chunks)
            store_in_chromadb(collection_name="localRAG_pdfs", texts=chunks)
            query = "What does the document say about the AI?"
            retrieved_texts = retrieve_from_chromadb(collection_name="localRAG_pdfs", query=query)
            response = generate_response_with_llm(retrieved_texts,query)
            print("\nTop Retrieved Chunks:")
            for r in retrieved_texts:
               print("--------", r)
            print("\nResponse:")
            print(response)
            #print(f"Stored {filename} in the database")
#if __name__ == "__main__":
#    main("data")