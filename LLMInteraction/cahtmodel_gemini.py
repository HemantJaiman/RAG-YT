from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model_name = "gemini-2.5-flash"
model = ChatGoogleGenerativeAI(model_name=model_name)

prompt = "Hello, how are you?"
response = model.invoke(prompt)
print(response)
    