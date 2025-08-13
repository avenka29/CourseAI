import chromadb
from langchain_chroma import Chroma
from google import genai

PERSIST_DIRECTORY = "./chroma_db_store"
GEMINI_API_KEY = "" # Replace with API key

chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
client = genai.Client(api_key=GEMINI_API_KEY)


question = input("Question: ")

collection = chroma_client.get_collection("CH101")
results = collection.query(
    query_texts=[question],
    n_results=5,
    where = {"course_id": {"$eq": "CH101"}}
)

retrieved_documents = results['documents'][0] if results['documents'] else []
retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else []

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