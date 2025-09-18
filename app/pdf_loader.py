from langchain.document_loaders import PyPDFLoader
from app.config import TEXT_SPLITTER

def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return TEXT_SPLITTER.split_documents(pages)