import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = "db"

def main():
    print("Ingestion script started...")

    # 1. Load documents
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="*.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    if not documents:
        print("No documents found to ingest. Exiting.")
        return

    # 2. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # 3. Initialize Ollama embeddings (latest package, device auto-handled)
    embeddings = OllamaEmbeddings(model="llama3")
    print("Embeddings model initialized.")

    # 4. Create Chroma vector store and persist
    print("Creating and persisting vector store...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print("Ingestion complete!")
    print(f"Vector store created at: {os.path.abspath(CHROMA_DB_DIR)}")

if __name__ == "__main__":
    main()
