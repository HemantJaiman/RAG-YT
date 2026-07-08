from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional, List


model = init_chat_model(
    model="llama3.1:8b",   
    model_provider="ollama",  
    base_url="http://localhost:11434",
    temperature=0.0,
    ) 
parser1 = StrOutputParser()

class Feedback(BaseModel):
    satisfaction: Literal["satisfied","neutral","dissatisfied"] = Field(description="Satisfaction level of the user")  
    suggestions: List[str] = Field(description="Suggestions for improvement", default=[])
    rating: Optional[int] = Field(description="Rating of the user", default=None)   

parser2 = PydanticOutputParser(pydantic_object=Feedback) 

prompt1 = PromptTemplate(
    template="""
    You are a helpful AI assistant. 
    Analyze the following user feedback  and determine whether the user is satisfied, neutral, or dissatisfied.
    check for suggestions for improvement.
    check for a rating.
    feedback: {feedback}
    provide the feedback in follwing format: {format_instructions}
    """,
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="""
    write a detailed reply to this positive feedback: {feedback} 
    """,
    input_variables=["feedback"],
)   

prompt3 = PromptTemplate(
    template="""
    write a detailed reply to this negative feedback: {feedback} 
    """,
    input_variables=["feedback"],
)       

prompt4 = PromptTemplate(
    template="""
    write a detailed reply to this neutral feedback: {feedback} 
    """,
    input_variables=["feedback"],   
)

# Create a dictionary runnable to preserve both the parsed feedback object AND the original raw input.
classification_and_input = {
    "parsed": classifier_chain,
    "input": RunnablePassthrough()
}

branch_chain = RunnableBranch(
    (lambda x: x["parsed"].satisfaction == "satisfied", (lambda x: x["input"]) | prompt2 | model | parser1), 
    (lambda x: x["parsed"].satisfaction == "dissatisfied", (lambda x: x["input"]) | prompt3 | model | parser1),
    (lambda x: x["parsed"].satisfaction == "neutral", (lambda x: x["input"]) | prompt4 | model | parser1),   
    (RunnableLambda(lambda x: "No valid sentiment is given by user"))    
)

pipeline = classification_and_input | branch_chain

print(pipeline.invoke({"feedback": "This is a good product. I really Liked this. I think its amazing"}))  

