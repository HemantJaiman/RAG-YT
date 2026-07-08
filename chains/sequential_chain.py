from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model(
    model="llama3.1:8b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    temperature=0.5,    
    )       

# sequential chain using LCEL
prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"],
    )
prompt2 = PromptTemplate(
    template="Write 3 points summary about {report}",
    input_variables=["report"],
    )

parser = StrOutputParser()
pipeline = prompt1 | model | parser | prompt2 | model | parser

result = pipeline.invoke({"topic": "Black hole"})
print(result)