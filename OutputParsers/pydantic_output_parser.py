from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage      

model = init_chat_model(model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.0
    )

class Person(BaseModel):
    Name: str = Field(description="Name of the person")
    Age: int = Field(gt=18, lt= 100, description="Age of the person")
    Occupation: str = Field(description="Occupation of the person") 
    City : str = Field(description="City of the person")   
    
parser = PydanticOutputParser(pydantic_object=Person)   


template = PromptTemplate(
    template="""
    Give me the details like Name, Age, Occupation, and City of the fictional person : {person}. make sure the age is above 18 and less then 100. 
    Also give me the output in the following format: {format_instructions}
    """,
    input_variables=["person"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)           

pipeline = template | model | parser

print(pipeline.invoke({"person": "John Doe"}))  