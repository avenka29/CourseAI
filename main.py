import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from google import genai

GEMINI_API_KEY = ""  # Replace with API key
client = genai.Client(api_key=GEMINI_API_KEY)


question = input("Question: ")
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


retrieved_docs = chroma_db.get()
collection = chroma_client.get_collection("CH101")
results = collection.query(
    query_texts=[question],
    n_results=5,
    where = {"course_id": {"$eq": "CH101"}}
)

# Get the retrieved documents
retrieved_documents = results['documents'][0] if results['documents'] else []
retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else []

# Prepare context for LLM
context_text = ""
for i, doc in enumerate(retrieved_documents):
    context_text += f"Document {i+1}:\n{doc}\n\n"

# Create prompt with context and query
user_query = question
prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from course materials.

Context:
{context_text}

Question: {user_query}

Please answer the question based on the context provided. If the information is not available in the context, please say
'I couldnt find that information in the syllabus.. Sorry!'. Be concise and accurate in your response.

Answer:"""

# Send to Gemini LLM
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents = prompt)
    print(f"User Query: {user_query}")
    print(f"Number of relevant documents found: {len(retrieved_documents)}")
    print(f"LLM Response: {response.text}")
except Exception as e:
    print(f"Error generating response: {str(e)}")
