from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field   
from typing import List, TypedDict, Annotated, Optional, Literal      

class Review(BaseModel):
    summary: str = Field(description="Brief Summary of the user input", alias="summary")
    sentiment: Literal["Positive","Negative","Neutral"] = Field(description="Sentiment of the user input", alias="sentiment")
    key_themes: List[str] = Field(description="Key themes of the user input", alias="key_themes")  
    pros: Optional[List[str]] = Field(description="Pros of the user input", alias="pros")
    cons: Optional[List[str]] = Field(description="Cons of the user input", alias="cons")       
    name: Optional[str] = Field(description="Write down the name of the reviewer.")

model = init_chat_model(model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.5
    )

# chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content= """
    You are a helpful AI assistant
    Analyze the following review and provide a structured summary
    
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{review}")   
])  

chat_history = [
    HumanMessage(content="Hi, my name is Alex."),
    AIMessage(content="Hello Alex! How can I help you today?")
]   

review = """
This review is written by Alex Mathew and he is a programmer at goggle
The iPhone 17 Pro Max is widely regarded as Apple’s most powerful smartphone, featuring a significant design overhaul with an aluminum unibody, a prominent camera "plateau," and the new A19 Pro chip.  It delivers category-leading battery life, up to 3,000 nits of peak brightness, and a versatile camera system with 8x optical zoom and an improved 18MP Center Stage front sensor.

Pros include superb performance, excellent video capture, and a bright, durable Ceramic Shield 2 display. However, cons involve the high starting price of $1,199, the aluminum frame feeling less durable than previous titanium models, and a heavy, large form factor. 

Early long-term users report mixed software experiences with iOS 26, citing issues with Safari responsiveness, dictation accuracy, and occasional system lag despite the powerful hardware.  While the hardware is exceptional, some critics argue the price gap over the standard iPhone 17 is hard to justify for non-professional users.
"""

prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "review": review,
    }   
)

response = model.with_structured_output(Review).invoke(prompt)
print(response) 

