import chromadb

from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./db/chroma_db")

collection_name = "my_stories"

collection = chroma_client.get_or_create_collection(
    collection_name, embedding_function=default_ef
)

# Define the documents

documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you?"},
    {"id": "doc3", "text": "What is your name?"},
    {
        "id": "doc4",
        "text": "Microservices are a great way to build scalable applications.",
    },
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# define the query
query_text = "Age of the Earth"

results = collection.query(query_texts=[query_text], n_results=3)

for idx, documents in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]
    print(
        f"For the query: {query_text}, \n the document with id: {doc_id} has a distance of: {distance}"
    )
