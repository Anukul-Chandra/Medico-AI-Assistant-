from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# step1: Load Raw Pdf

file_path = "Data/"

def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader) 
    documents = loader.load()
    return documents

documents = load_pdf(file_path)

print(f"Total PDF Pages Loaded: {len(documents)}")


if documents:
    # strip function will remove the whitespace
    content = documents[0].page_content.strip() 
    
    # if 1st page empty then will check for 2nd page
    if not content:
        content = documents[1].page_content.strip() 

    print(f"Preview: {content[:200]}...") 
print("Congrats! My code is running perfectly.")


# step2: create chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"Total Chunks Created: {len(text_chunks)}")


# step3: create vector embeddings

def get_embedding_model():
    # Fixed: Changed 'HuggingfaceEmbeddings' to 'HuggingFaceEmbeddings' (Capital F)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()
print("Embedding model loaded successfully.")


# step4: store in vector embedding in FAISS

DB_PATH = "vectorstore/faiss_db"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_PATH)
print(f"Vector store saved at: {DB_PATH}")