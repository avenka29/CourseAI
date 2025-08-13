import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./chroma_db_store"

DOCUMENTS_TO_PROCESS = [
    {
        "path": "/sample.pdf",
        "content_for_dummy": "This is the syllabus for CH101, a chemistry class for engineering students",
        "metadata": {
            "course_id": "CH101",
            "course_name": "Chemistry: A Molecular Science ",
            "file_type": "syllabus",
            "year": "2025"
        }
    }
]


chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

for doc in docs:
    doc.metadata.update(DOCUMENTS_TO_PROCESS[0]["metadata"])

chroma_db = Chroma(
    collection_name="CH101",
    embedding_function=embeddings,
    client=chroma_client
)
chroma_db.add_documents(docs)

print("Documents added")

