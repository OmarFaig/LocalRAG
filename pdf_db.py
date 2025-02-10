import os
from PyPDF2 import PdfReader
import chromadb

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
            
            content += page.extract_text()
        return content
    
def store_in_chromadb(file_path, content):
    collection = client.get_or_create_collection("pdf_collection")
    document = {
        "id": os.path.basename(file_path),
        "content": content
    }
    collection.upsert(document)

def main(pdf_dir):
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            content = read_file(file_path)
            store_in_chromadb(file_path, content)
            print(f"Stored {filename} in the database")
if __name__ == "__main__":
    main("data")