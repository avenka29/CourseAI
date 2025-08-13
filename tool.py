from langchain.agents import tool
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import get_google_credentials, build_resource_service
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os as os


PERSIST_DIRECTORY = "./chroma_db_store"
GEMINI_API_KEY = "" # Replace with API key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
chroma_db_collection = Chroma(
    collection_name="CH101",
    embedding_function=embeddings,
    client=chroma_client
)

@tool
def retrieve_syllabus_info(question: str) -> str:
    """A tool for retrieving information from the CH101 course syllabus.
    Use this tool to answer questions about the course, deadlines, topics, and schedule.
    The input should be a precise question about the course content."""

    results = chroma_db_collection.similarity_search(question)
    
    retrieved_documents = [doc.page_content for doc in results]
    
    context_text = ""
    for i, doc in enumerate(retrieved_documents):
        context_text += f"Document {i+1}:\n{doc}\n\n"
    return context_text

credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)

api_resource = build_resource_service(credentials=credentials)
calendar_toolkit = CalendarToolkit(api_resource=api_resource)
calendar_tools = calendar_toolkit.get_tools()

all_tools = [retrieve_syllabus_info] + calendar_tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, all_tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

question = input("Question : ")
response = agent_executor.invoke({
    "input": question + "timezone = EST"
})

print(response)