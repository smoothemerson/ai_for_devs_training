from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2:3b")
embeddings_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")
