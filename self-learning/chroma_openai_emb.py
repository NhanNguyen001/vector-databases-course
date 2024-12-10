import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

default_ef = embedding_functions.DefaultEmbeddingFunction()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=open_api_key,
    model_name="text-embedding-3-small",
)

chroma_client = chromadb.PersistentClient(path="./db/chroma_openai_db")

collection_name = "my_stories"

collection = chroma_client.get_or_create_collection(
    collection_name, embedding_function=openai_ef
)


# Load documents from JSON file
def load_documents():
    try:
        with open("documents.json", "r") as file:
            data = json.load(file)
            return data["documents"]
    except FileNotFoundError:
        print("Error: documents.json file not found!")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in documents.json!")
        return []


# Load and insert documents
documents = load_documents()
if documents:
    print(f"\nLoaded {len(documents)} documents from JSON file")
    for doc in documents:
        collection.upsert(ids=doc["id"], documents=[doc["text"]])

# Check collection count
print(f"Total documents in collection: {collection.count()}")

# define the query
query_text = "find document related to technology company"

results = collection.query(query_texts=[query_text], n_results=3)


def main():
    print(f"\nQuery: {query_text}")
    print("\nResults:")
    for idx, document in enumerate(results["documents"][0]):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        print(f"\nDocument {idx + 1}:")
        print(f"Content: {document}")
        print(f"ID: {doc_id}")
        print(f"Distance: {distance}")


if __name__ == "__main__":
    main()
