from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFace sentence transformer model (compact and good)
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chunking settings
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Directory where FAISS stores vectors
FAISS_DIR = "faiss_index"
