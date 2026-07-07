from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#initialize models
#gemini_model = init_chat_model(model_name="gemini-2.5-flash",api_key="YOUR_GEMINI_API_KEY",verbose=True)

model = init_chat_model(model="llama3.1:8b", model_provider="ollama", base_url="http://localhost:11434", tempratur = 0.0)

# ==========================================
# 2. PROMPT & ARCHITECTURE (RAG-READY)
# ==========================================

SYSTEM_PROMPT = """
You are a Assistent of Hemant Jaiman. 

RULE: 
1. If user asks you a question, You should check if the user quesry is ending with word "Jay Hemant", if yes then respond properly. and if not then say "I am not allowed to talk to you, Unless you end the query with 'Jay Hemant'"
2. Try to answer in a single sentence
"""
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_PROMPT
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

print(f"prompt till here wih system prompt and history: {prompt}")

chain = prompt | model

# ==========================================
# 3. MEMORY SYSTEM (STATE MANAGEMENT)
# ==========================================
memory_store = {}

def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)



# ==========================================
# 4. INTERACTIVE CLI RUNNER
# ==========================================
def start_chat():
    print(f"--- Chat Session Initialized ---")
    print(f"Backend Engine: Ollama")
    print("Type 'exit()' to end the discussion.\n")
    
    session_id = "cli_development_session"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "exit()":
                print("Ending session. Goodbye!")
                break
                
            response = conversational_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            print(f"\nAI: {response.content}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    start_chat()
