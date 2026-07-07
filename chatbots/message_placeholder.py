from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model

# 1. Initialize the Model 
model = init_chat_model(
    model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.0
)

# 2. Chat Template 
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content= """
        You are a helpful AI assistant
        Answer the user's questions politely and informatively
        Be concise and clear
        If you don't know the answer, say so
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 3. Load Chat History 
chat_history = [
    HumanMessage(content="Hi, my name is Alex."),
    AIMessage(content="Hello Alex! How can I help you today?")
]

# 4. Compile the Prompt
prompt = chat_template.invoke({
    "chat_history": chat_history, 
    "question": "What is your name?"
})

print(prompt)


response = model.invoke(prompt)
print(response.content)