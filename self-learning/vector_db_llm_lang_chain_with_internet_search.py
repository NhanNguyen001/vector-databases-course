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

    # Check if this is a version-related query
    is_version_query = any(
        keyword in query.lower() for keyword in ["version", "latest", "recent"]
    )

    # For version queries, always check web first
    if is_version_query:
        try:
            search = DuckDuckGoSearchRun()
            search_results = search.run(f"latest version {query}")
            if search_results.strip():
                web_doc = Document(
                    page_content=search_results,
                    metadata={
                        "source": "web_search (current)",
                        "date": "current",
                        "query": query,
                    },
                )
                all_docs.insert(0, web_doc)
                source_info.append("web search")
                print("Retrieved latest version information from web search.")

                # Store in vector database to update local information
                print("Updating version information in database...")
                vectordb.add_documents([web_doc])
                print("Successfully updated version information in database.")
        except Exception as e:
            print(f"Error searching the internet: {e}")

    # Then check local database
    try:
        local_docs = vectordb.similarity_search(query, k=3)
        if local_docs:
            # For version queries, only use local docs as fallback
            if not is_version_query or not all_docs:
                all_docs.extend(local_docs)
                source_info.append("local database")
                print("Found relevant documents in local database.")
        else:
            print("No relevant documents found in local database.")

            # Only search internet for non-version queries if nothing found locally
            if not is_version_query:
                try:
                    search = DuckDuckGoSearchRun()
                    search_results = search.run(query)
                    if search_results.strip():
                        web_doc = Document(
                            page_content=search_results,
                            metadata={
                                "source": "web_search (current)",
                                "date": "current",
                                "query": query,
                            },
                        )
                        all_docs.insert(0, web_doc)
                        source_info.append("web search")
                        print("Retrieved information from web search.")

                        # Store in vector database
                        print("Storing new web search result in database...")
                        vectordb.add_documents([web_doc])
                        print("Successfully stored new content in database.")
                except Exception as e:
                    print(f"Error searching the internet: {e}")
    except Exception as e:
        print(f"Error retrieving local documents: {e}")

    if not all_docs:
        return "I apologize, but I couldn't find any relevant information from either local documents or web search."

    # Create the QA chain
    chain = prompt | model | StrOutputParser()

    # Format the context with clear source and date information
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')}, Date: {doc.metadata.get('date', 'unknown')}, Query: {doc.metadata.get('query', 'unknown')}]\n{doc.page_content}"
            for doc in all_docs
        ]
    )

    # Enhance the query to emphasize current version information for version queries
    if is_version_query:
        enhanced_query = f"{query}\n\nPlease note: This information is sourced from {', '.join(source_info)}. Focus on the most recent version number and release date. If multiple versions are mentioned, specify the latest one. Start your response with '[Source: {', '.join(source_info)}]'."
    else:
        enhanced_query = f"{query}\n\nPlease note: This information is sourced from {', '.join(source_info)}. Prioritize the most recent information and explicitly mention the date of the information if available. Start your response with '[Source: {', '.join(source_info)}]'."

    response = chain.invoke({"context": context, "input": enhanced_query})
    return response


# Example usage
queries = [
    "Which latst version of FastAPI framework?",
    # "Which latst version of Databricks Runtime?",
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
