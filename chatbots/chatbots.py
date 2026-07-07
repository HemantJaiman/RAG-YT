from langchain_core import chat_history
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(model="llama3.1:8b", model_provider="ollama", base_url="http://localhost:11434", tempratur = 0.0)

SYSTEM_PROMPT = "You are a helpful AI assistant"

chat_history = [
    SystemMessage(content=SYSTEM_PROMPT)
]

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ", response.content) 

print(f"chat History: {chat_history}")