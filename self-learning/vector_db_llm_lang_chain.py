import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")

# Load the vector database
loader = DirectoryLoader(
    path="./data/new_articles", glob="*.txt", loader_cls=TextLoader
)

document = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=20
)

documents = text_splitter.split_documents(document)
print(f"Number of documents: {len(documents)}")

embedding = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")

persist_directory = "./db/chroma_db_real_world"
vectordb = Chroma.from_documents(
    documents=documents, embedding=embedding, persist_directory=persist_directory
)

print(vectordb)
