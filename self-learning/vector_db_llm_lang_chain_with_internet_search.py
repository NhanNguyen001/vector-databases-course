import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

system_prompt = (
    "You are a knowledgeable assistant that provides accurate and up-to-date information. "
    "Use the following pieces of retrieved context to answer the question. "
    "If information is found from multiple sources (local documents and web), "
    "combine them to provide a comprehensive answer. "
    "Always prioritize recent information. "
    "If the information seems outdated, mention that in your response. "
    "Keep the answer concise but informative."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

openai_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(api_key=openai_key, model="gpt-4o")

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


def get_answer(query: str):
    all_docs = []
    source_info = []

    # First, try to get answer from local vector database
    try:
        # Using similarity search with MMR for better diversity in results
        local_docs = vectordb.max_marginal_relevance_search(query, k=3, fetch_k=10)
        if local_docs:
            all_docs.extend(local_docs)
            source_info.append("local database")
            print("Found relevant documents in local database.")
    except Exception as e:
        print(f"Error retrieving local documents: {e}")

    # Always search the internet to get the most recent information
    try:
        search = DuckDuckGoSearchRun()
        search_results = search.run(query)
        if search_results.strip():
            web_doc = Document(
                page_content=search_results,
                metadata={"source": "web_search", "date": "current"},
            )
            all_docs.append(web_doc)
            source_info.append("web search")
            print("Retrieved information from web search.")
    except Exception as e:
        print(f"Error searching the internet: {e}")

    if not all_docs:
        return "I apologize, but I couldn't find any relevant information from either local documents or web search."

    # Create the QA chain
    chain = prompt | model | StrOutputParser()

    # Format the context and query
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in all_docs
        ]
    )

    # Add source information to the query
    enhanced_query = f"{query} (Information sourced from: {', '.join(source_info)})"

    response = chain.invoke({"context": context, "input": enhanced_query})
    return response


# Example usage
queries = [
    "Which latst version of Databricks Runtime?",
    # "Talk about databricks new?",
    # "What are the latest developments in AI?",
    # "Tell me about recent machine learning frameworks",
]

for query in queries:
    print(f"\nQuestion: {query}")
    print("-" * 40)
    answer = get_answer(query)
    print(f"Answer: {answer}")
    print("-" * 80)
