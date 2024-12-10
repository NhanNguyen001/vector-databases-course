from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

response = client.embeddings.create(
    input="Hello, world!",
    model="text-embedding-3-small",
)

print(response.data[0].embedding)