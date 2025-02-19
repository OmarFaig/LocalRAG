import os
from PyPDF2 import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from llama_cpp import Llama
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Initialize LangChain components
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(embedding_model=embedding_model, collection_name="localRAG_pdfs")
llm = Llama(model_path="llama-2-7b-chat-codeCherryPop.Q3_K_S.gguf")

def read_file(filepath):
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        content = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            content += page.extract_text() + "\n"
        return content

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@app.post("/upload/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    file_urls = []
    for file in files:
        try:
            file_location = f"data/{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
            content = read_file(file_location)
            chunks = chunk_text(content)
            vector_store.add_texts(chunks)
            file_urls.append({"filename": file.filename, "file_url": f"/data/{file.filename}"})
        except Exception as e:
            return {"error": str(e)}
    return {"file_urls": file_urls}

@app.post("/query/")
async def query_rag(query: str = Form(...)):
    results = vector_store.similarity_search(query, k=3)
    retrieved_texts = [result["text"] for result in results]
    context = "\n".join(retrieved_texts)[:400]
    prompt = f"Context: {context}\nQuery: {query}\nResponse:"
    response = llm(prompt, max_tokens=100)
    return {"response": response["choices"][0]["text"].strip()}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)