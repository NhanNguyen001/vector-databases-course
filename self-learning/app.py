import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(
    collection_name, embedding_function=default_ef
)

# Define the documents
documents = [
    {"id": "doc1", "text": "Hello, worl"},
    {"id": "doc2", "text": "How are you?"},
    {"id": "doc3", "text": "What is your name?"},
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# define the query
query_text = "Hello, world!"

results = collection.query(query_texts=[query_text], n_results=3)

print(results)

for idx, documents in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(
        f"For the query: {query_text}, the document with id: {doc_id} has a distance of: {distance}"
    )
