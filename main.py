import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings



DOCUMENTS_TO_PROCESS = [
    {
        "path": "/sample.pdf",
        "content_for_dummy": "This is the syllabus for CH101, a chemistry class for engineering students",
        "metadata": {
            "course_id": "CH101",
            "course_name": "Chemistry: A Molecular Science ",
            "file_type": "syllabus",
            "year": "2023"
        }
    }
]


chroma_client = chromadb.Client()

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Tag and store
for doc in docs:
    doc.metadata.update(DOCUMENTS_TO_PROCESS[0]["metadata"])

chroma_db = Chroma(
    collection_name="CH101",
    embedding_function=embeddings,
    client=chroma_client
)
chroma_db.add_documents(docs)


print("\n--- Stored Documents ---")
retrieved_docs = chroma_db.get()
collection = chroma_client.get_collection("CH101")
results = collection.query(
    query_texts=["When is the final exam?"],
    n_results=5,
    where = {"course_id": {"$eq": "CH101"}}
)

documents = results.get('documents', [])
metadatas = results.get('metadatas', [])

for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
    print(f"Result {i}:")
    print("Content:", doc[:300], "...")  # print first 300 chars
    print("Metadata:", meta)
    print("-" * 40)
