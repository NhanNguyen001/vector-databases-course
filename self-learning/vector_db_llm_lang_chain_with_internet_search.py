import os
import json
import requests
import re
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import chromadb
from packaging import version

# Suppress SSL verification warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


load_dotenv()


def extract_package_name(query):
    """Extract potential package name from version query."""
    # Common patterns in version queries
    patterns = [
        r"latest version of (\w+)",
        r"(\w+) latest version",
        r"(\w+) version",
        r"version of (\w+)",
        r"latst version of (\w+)",  # Common typo
        r"(\w+) latst version",  # Common typo
    ]

    query = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)

    # If no pattern matches, try to find framework names directly
    frameworks = {
        "fastapi": ["fastapi", "fast-api", "fast api"],
        "django": ["django"],
        "flask": ["flask"],
        # Add more frameworks as needed
    }

    for framework, variants in frameworks.items():
        if any(variant in query.lower() for variant in variants):
            return framework

    return None


system_prompt = (
    "You are a knowledgeable assistant that provides accurate and up-to-date information. "
    "Use the following pieces of retrieved context to answer the question. "
    "If information is found from multiple sources (local documents and web), "
    "combine them to provide a comprehensive answer. "
    "For version-related queries:"
    "1. Compare version numbers from different sources"
    "2. Only report a version if it's confirmed by multiple sources"
    "3. Include the release date if available"
    "4. If sources conflict, mention the discrepancy"
    "5. If the version information seems outdated, explicitly mention that"
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

# Initialize the ChromaDB client
chroma_client = chromadb.PersistentClient(path=persist_directory)

# Initialize or load the vector database
try:
    # Try to get or create the collection
    collection_name = "document_store"
    print("Loading existing vector database...")
    vectordb = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding,
    )

    # If it's a new database, add initial documents
    if vectordb._collection.count() == 0:
        print("Adding initial documents to database...")
        vectordb.add_documents(documents)
except Exception as e:
    print(f"Error initializing database: {e}")
    print("Creating new vector database...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def extract_version_info(text):
    """Extract version numbers and dates from text."""
    version_pattern = r"(?:version|v)?\.?\s*(\d+\.\d+\.\d+(?:-?\w+)?)"
    date_pattern = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+ \d{1,2},? \d{4})"

    versions = re.findall(version_pattern, text, re.IGNORECASE)
    dates = re.findall(date_pattern, text)

    return versions, dates


def validate_version_freshness(content):
    """Validate if the version information is recent and consistent."""
    versions, dates = extract_version_info(content)

    if not versions:
        return False, "No version information found"

    # Convert versions to tuples of integers for comparison
    version_tuples = []
    for v in versions:
        try:
            # Remove any pre-release suffixes for comparison
            clean_version = re.match(r"(\d+\.\d+\.\d+)", v).group(1)
            version_parts = [int(x) for x in clean_version.split(".")]
            version_tuples.append((version_parts, v))
        except (AttributeError, ValueError):
            continue

    if not version_tuples:
        return False, "Could not parse version numbers"

    # Sort versions and get the latest
    version_tuples.sort(reverse=True)
    latest_version = version_tuples[0][1]

    # Check if we have multiple sources with the same version
    version_count = sum(1 for v in version_tuples if v[0] == version_tuples[0][0])

    if version_count < 2:
        return (
            False,
            f"Version {latest_version} found but not confirmed by multiple sources",
        )

    return True, f"Latest version {latest_version} confirmed by multiple sources"


def get_fastapi_version():
    try:
        response = requests.get("https://pypi.org/pypi/fastapi/json")
        if response.status_code == 200:
            data = response.json()
            version = data["info"]["version"]
            release_date = list(data["releases"][version])[-1]["upload_time"].split(
                "T"
            )[0]
            return f"The latest version of FastAPI is {version}, released on {release_date}"
    except Exception as e:
        print(f"Error fetching FastAPI version: {e}")
    return None


def get_web_content(query):
    """Get information from web sources with validation."""
    try:
        # First attempt with regular search
        search_urls = [
            f"https://www.google.com/search?q={query}+latest+version+release",
            f"https://www.bing.com/search?q={query}+latest+version+release",
            f"https://duckduckgo.com/html?q={query}+latest+version+release",
        ]

        loader = WebBaseLoader(
            search_urls,
            verify_ssl=False,
            header_template={"User-Agent": os.environ.get("USER_AGENT")},
        )
        docs = loader.load()

        # Combine all web content
        combined_content = "\n".join([doc.page_content for doc in docs])

        # Validate version information
        is_valid, message = validate_version_freshness(combined_content)

        if not is_valid:
            print(f"Initial validation: {message}")
            print("Checking additional sources...")

            # Try additional sources with different search patterns
            additional_urls = [
                f"https://www.google.com/search?q={query}+changelog+latest",
                f"https://www.bing.com/search?q={query}+release+notes+latest",
                f"https://duckduckgo.com/html?q={query}+github+latest+release",
            ]

            additional_loader = WebBaseLoader(
                additional_urls,
                verify_ssl=False,
                header_template={"User-Agent": os.environ.get("USER_AGENT")},
            )
            additional_docs = additional_loader.load()

            # Combine with original content
            combined_content += "\n" + "\n".join(
                [doc.page_content for doc in additional_docs]
            )

            # Revalidate with combined content
            is_valid, message = validate_version_freshness(combined_content)
            print(f"Final validation: {message}")

        return combined_content
    except Exception as e:
        print(f"Error fetching web content: {e}")
        return None


def get_pypi_version(package_name):
    """Get the latest version information for any Python package from PyPI."""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        if response.status_code == 200:
            data = response.json()
            releases = data["releases"]

            # Filter out pre-releases and get all version numbers
            stable_versions = []
            for ver, release_info in releases.items():
                try:
                    parsed_version = version.parse(ver)
                    # Skip pre-releases
                    if not parsed_version.is_prerelease and release_info:
                        stable_versions.append((ver, release_info[-1]["upload_time"]))
                except version.InvalidVersion:
                    continue

            if stable_versions:
                # Sort by version number (using packaging.version) and upload time
                stable_versions.sort(
                    key=lambda x: (version.parse(x[0]), x[1]), reverse=True
                )
                latest_version, upload_time = stable_versions[0]
                formatted_date = upload_time.split("T")[0]
                return f"The latest stable version of {package_name} is {latest_version}, released on {formatted_date}"

    except Exception as e:
        print(f"Error fetching {package_name} version: {e}")
    return None


def get_answer(query: str):
    all_docs = []
    source_info = []

    # Check if this is a version-related query
    is_version_query = any(
        keyword in query.lower()
        for keyword in ["version", "latest", "recent", "lastest"]
    )

    # First check local database
    try:
        local_docs = vectordb.similarity_search(query, k=3)
        if local_docs:
            all_docs.extend(local_docs)
            source_info.append("local database")
            print("Found relevant documents in local database.")

            # Check if the local documents actually contain version information
            has_version_info = any(
                "version" in doc.page_content.lower()
                and any(
                    num in doc.page_content
                    for num in ["0.", "1.", "2.", "3.", "4.", "5."]
                )
                for doc in local_docs
            )

            if not has_version_info and is_version_query:
                print(
                    "Local documents don't contain specific version information. Searching online..."
                )
                all_docs = []  # Clear local docs to use online info instead
                source_info = []
        else:
            print("No relevant documents found in local database. Searching online...")
    except Exception as e:
        print(f"Error retrieving local documents: {e}")

    # If no relevant local docs or no version info found, search online
    if not all_docs and is_version_query:
        try:
            web_content = get_web_content(query)
            if web_content:
                web_doc = Document(
                    page_content=web_content,
                    metadata={
                        "source": "Web Search (current)",
                        "date": "current",
                        "query": query,
                    },
                )
                all_docs.append(web_doc)
                source_info.append("Web Search")
                print("Retrieved information from web search.")

                # Store in vector database
                print("Storing web search result in database...")
                vectordb.add_documents(
                    [web_doc], ids=[f"web_search_{os.urandom(4).hex()}"]
                )
                print("Successfully stored new content in database.")
        except Exception as e:
            print(f"Error searching the internet: {e}")

    if not all_docs:
        return "I apologize, but I couldn't find any relevant information from either local documents or web search."

    # Create the QA chain
    chain = prompt | model | StrOutputParser()

    # Format the context with clear source and date information
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')}, Date: {doc.metadata.get('date', 'unknown')}]\n{doc.page_content}"
            for doc in all_docs
        ]
    )

    # Enhance the query to emphasize current version information for version queries
    if is_version_query:
        enhanced_query = f"{query}\n\nPlease note: This information is sourced from {', '.join(source_info)}. Focus on the most recent version number and release date. If multiple versions are mentioned, specify the latest one."
    else:
        enhanced_query = f"{query}\n\nPlease note: This information is sourced from {', '.join(source_info)}. Prioritize the most recent information and explicitly mention the date of the information if available."

    response = chain.invoke({"context": context, "input": enhanced_query})
    return response


# Example usage
queries = [
    "Which latest version of FastAPI python package?",
    "Which latest version of Uvicorn python package?",
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
