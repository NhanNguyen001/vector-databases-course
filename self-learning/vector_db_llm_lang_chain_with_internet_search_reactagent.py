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
from packaging import version
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.tools import tool

# Suppress SSL verification warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

load_dotenv()

# Initialize OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(api_key=openai_key, model="gpt-4")

# Initialize text splitter for web content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
)


def extract_version_info(text):
    """Extract version numbers and dates from text."""
    version_pattern = r"(?:version|v)?\.?\s*(\d+\.\d+\.\d+(?:-?\w+)?)"
    date_pattern = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+ \d{1,2},? \d{4})"

    versions = re.findall(version_pattern, text, re.IGNORECASE)
    dates = re.findall(date_pattern, text)

    return versions, dates


@tool
def search_web_version(query: str) -> str:
    """Search the web for version information about a software package."""
    try:
        search_urls = [
            f"https://www.google.com/search?q={query}+latest+version+release",
            f"https://www.bing.com/search?q={query}+latest+version+release",
            f"https://duckduckgo.com/html?q={query}+latest+version+release",
        ]

        loader = WebBaseLoader(
        loader = WebBaseLoader(
            search_urls,
            verify_ssl=False,
            header_template={"User-Agent": os.environ.get("USER_AGENT")},
        )
        docs = loader.load()

        # Split long texts into smaller chunks
        all_content = "\n".join([doc.page_content for doc in docs])
        chunks = text_splitter.split_text(all_content)

        # Return only the most relevant first chunk
        return chunks[0] if chunks else ""
    except Exception as e:
        return f"Error searching web: {str(e)}"


@tool
def search_changelog(query: str) -> str:
    """Search for changelog or release notes of a software package."""
    try:
        search_urls = [
            f"https://www.google.com/search?q={query}+changelog+latest+release",
            f"https://www.bing.com/search?q={query}+release+notes+latest",
            f"https://duckduckgo.com/html?q={query}+github+latest+release",
        ]

        loader = WebBaseLoader(
            search_urls,
            verify_ssl=False,
            header_template={"User-Agent": os.environ.get("USER_AGENT")},
        )
        docs = loader.load()

        # Split long texts into smaller chunks
        all_content = "\n".join([doc.page_content for doc in docs])
        chunks = text_splitter.split_text(all_content)

        # Return only the most relevant first chunk
        return chunks[0] if chunks else ""
    except Exception as e:
        return f"Error searching changelog: {str(e)}"


@tool
def extract_and_validate_versions(text: str) -> str:
    """Extract and validate version numbers from text."""
    versions, dates = extract_version_info(text)

    if not versions:
        return "No version information found"

    # Convert versions to tuples for comparison
    version_tuples = []
    for v in versions:
        try:
            clean_version = re.match(r"(\d+\.\d+\.\d+)", v)
            if clean_version:
                version_parts = [int(x) for x in clean_version.group(1).split(".")]
                version_tuples.append((version_parts, v, clean_version.group(1)))
        except (AttributeError, ValueError):
            continue

    if not version_tuples:
        return "Could not parse version numbers"

    # Sort versions
    version_tuples.sort(reverse=True)
    latest_version = version_tuples[0][2]

    # Count occurrences of the latest version
    version_count = sum(1 for v in version_tuples if v[0] == version_tuples[0][0])

    if version_count >= 2:
        if dates:
            dates.sort(reverse=True)
            return f"Latest version {latest_version} confirmed by multiple sources, most recent date: {dates[0]}"
        return f"Latest version {latest_version} confirmed by multiple sources"

    return (
        f"Found version {latest_version} but needs confirmation from additional sources"
    )


# Create the ReAct agent prompt
react_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a version checking agent that finds and validates the latest versions of software packages.

Available Tools:
{tools}

Tool Names: {tool_names}

To find the latest version of a package, follow these steps carefully:

1. First, use search_web_version to find initial version information
   - Input: The package name and "latest version"
   - This will search multiple sources

2. Use extract_and_validate_versions on the search results
   - Input: The text from the search
   - This will find and validate version numbers

3. If needed, use search_changelog to confirm the version
   - Input: The package name and "changelog" or "release notes"
   - This helps verify the version information

4. Use extract_and_validate_versions again on the changelog
   - This ensures version consistency across sources

5. Compare all findings and report:
   - The latest confirmed version
   - The release date if available
   - Any discrepancies found
   - Confidence in the information

To use a tool, follow this format exactly:
Thought: what you're thinking about doing
Action: the_tool_name
Action Input: the input to the tool
Observation: the result of the tool
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}""",
        ),
    ]
)

# Create the tools list
tools = [
    Tool(
        name="search_web_version",
        func=search_web_version,
        description="Search the web for version information about a software package. Input should be the package name and 'latest version'.",
    ),
    Tool(
        name="search_changelog",
        func=search_changelog,
        description="Search for changelog or release notes of a software package. Input should be the package name and 'changelog' or 'release notes'.",
    ),
    Tool(
        name="extract_and_validate_versions",
        func=extract_and_validate_versions,
        description="Extract and validate version numbers from text. Input should be the text containing version information.",
    ),
]

# Create the agent
agent = create_react_agent(
    llm=ChatOpenAI(temperature=0), tools=tools, prompt=react_prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create the QA prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledgeable assistant that provides accurate and up-to-date information. 
    Use the following pieces of retrieved context to answer the question. 
    If information is found from multiple sources, combine them to provide a comprehensive answer. 
    For version-related queries:
    1. Compare version numbers from different sources
    2. Only report a version if it's confirmed
    3. Include the release date if available
    4. If sources conflict, mention the discrepancy
    5. If the version information seems outdated, explicitly mention that
    
    Context: {context}""",
        ),
        ("human", "{input}"),
    ]
)


def get_answer(query: str):
    all_docs = []
    source_info = []

    # Check if this is a version-related query
    is_version_query = any(
        keyword in query.lower()
        for keyword in ["version", "latest", "recent", "lastest"]
    )

    if is_version_query:
        try:
            # Use the ReAct agent for version queries
            result = agent_executor.invoke({"input": query})
            web_doc = Document(
                page_content=result["output"],
                metadata={
                    "source": "ReAct Agent",
                    "date": "current",
                    "query": query,
                },
            )
            all_docs.append(web_doc)
            source_info.append("ReAct Agent")

            # Store in vector database
            print("Storing version information in database...")
            vectordb.add_documents(
                [web_doc], ids=[f"react_agent_{os.urandom(4).hex()}"]
            )
        except Exception as e:
            print(f"Error in ReAct agent: {e}")

    # Check local database as fallback
    try:
        local_docs = vectordb.similarity_search(query, k=3)
        if local_docs:
            all_docs.extend(local_docs)
            source_info.append("local database")
            print("Found relevant documents in local database.")
    except Exception as e:
        print(f"Error retrieving local documents: {e}")

    if not all_docs:
        return "I apologize, but I couldn't find any relevant information from either the ReAct agent or local database."

    # Create the QA chain
    chain = qa_prompt | model | StrOutputParser()

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
if __name__ == "__main__":
    # Initialize vector database
    loader = DirectoryLoader(
        path="./data/new_articles", glob="*.txt", loader_cls=TextLoader
    )
    document = loader.load()

    documents = text_splitter.split_documents(document)
    print(f"Number of documents: {len(documents)}")

    embedding = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")
    persist_directory = "./db/chroma_db_real_world"

    # Initialize or load the vector database
    try:
        print("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )

        # If it's a new database, add initial documents
        if len(vectordb.get()["ids"]) == 0:
            print("Adding initial documents to database...")
            vectordb.add_documents(documents)
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Creating new vector database...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory,
        )

    # Run queries
    queries = [
        "Which latest version of FastAPI framework?",
        "Which latest version of Uvicorn python package?",
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        print("-" * 40)
        answer = get_answer(query)
        print(f"Answer: {answer}")
        print("-" * 80)
