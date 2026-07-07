from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict


class Review(TypedDict):
    Summary: str
    Sentiment: str
    
# 1. Initialize the Model 
model = init_chat_model(
    model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.0
)

structured_model = model.with_structured_output(Review)  


# 2. creating chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content= """
    You are a helpful AI assistant
    Answer the user's questions politely and informatively
    Be concise and clear
    If you don't know the answer, say so
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human"   , "{question}")
])

# 3. creating chat history
chat_history = [
    HumanMessage(content="Hi, my name is Alex."),
    AIMessage(content="Hello Alex! How can I help you today?")
]   

# 4. compiling the prompt
prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "question": "I really like this moview, i think it looks like full of emotions. i enjoyed this movie very much.",
    }
)

# 5. invoking the model
response = structured_model.invoke(prompt)

print(response)
