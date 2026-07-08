from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field   
from typing import List, TypedDict, Annotated, Optional, Literal     
from langchain_core.runnables import RunnablePassthrough, RunnableParallel 

model = init_chat_model(model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.5
    )   

prompt1 = PromptTemplate(
    template="""
    Analyze the following text and provide a structured summary about it: {text}
    
    """,
    input_variables=["text"],
    ) 
prompt2 = PromptTemplate(
    template="""
    Based on the following text : {text}, generate 5 Q&A pairs.
    """,
    input_variables=["text"],   
    )       
prompt3 = PromptTemplate(
    template="""
    merge these 2 points togather and properly generate a final summary and small quiz at the end
    summary: {summary}
    Q&A : {qa}
    
    """,
    input_variables=["summary", "qa"],   
    )       

parser = StrOutputParser()

runnable1 = RunnableParallel(
    summary = (prompt1 | model | parser),
    qa = (prompt2 | model | parser),
    )

runnable = runnable1 | prompt3 | model | parser

result = runnable.invoke({"text": "black hole"})
print(result)
