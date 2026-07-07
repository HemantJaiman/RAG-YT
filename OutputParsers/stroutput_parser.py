from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

model = init_chat_model(model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.0
    )

template1 = PromptTemplate(
    template="""
    wrtie a detailed report on {topic}
    """,
    input_variables=["topic"],
    )   
template2 = PromptTemplate(
    template = "write a 5 points summary on this following {text}",
    input_variables=["text"],
)   


parser = StrOutputParser()

pipeline = template1 | model | parser | template2 | model | parser

print(pipeline.invoke("Indian premier league 2025/2026"))