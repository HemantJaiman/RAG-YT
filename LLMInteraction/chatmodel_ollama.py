from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")

prompt = "Hello, how are you?"
response = llm.invoke(prompt)
print(response)

print("here is the answer: ", response.content)
