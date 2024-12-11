import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")

messages = [
    SystemMessage("Translate the following from English into Spanish"),
    HumanMessage("hi!"),
]

response = model.invoke(messages)

print(response.content)
