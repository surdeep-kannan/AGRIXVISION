import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Configuration ---
CHROMA_DB_DIR = "db"

# 1. Load embeddings
embeddings = OllamaEmbeddings(model="llama3")

# 2. Load vector store
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# 3. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Initialize LLM
llm = OllamaLLM(model="llama3")

# 5. Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 6. Interactive QA loop
print("You can now ask questions about your knowledge base. Type 'exit' to quit.")
while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        print("Exiting...")
        break
    try:
        answer = qa_chain.run(query)
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure the Ollama server is running (e.g., `ollama run llama3`).")

