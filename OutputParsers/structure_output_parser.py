from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model


# 1. Initialize the Model
model = init_chat_model(
    model="llama3.1:8b", 
    model_provider="ollama", 
    base_url="http://localhost:11434", 
    temperature=0.0
)

# 2. Define schema using Pydantic V2
class TopicReport(BaseModel):
    title: str = Field(description="The main title of the topic")
    summary: str = Field(description="A concise summary of the topic")
    key_points: list[str] = Field(description="A list of 3 key takeaways or facts about the topic")


# =====================================================================
# METHOD 1: The Standard Modern LCEL Way (No self-correction loop needed)
# =====================================================================
# In modern LangChain, using `with_structured_output` inside an LCEL chain (with `|`)
# is the standard and easiest way to enforce structured output.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Output a structured report about the topic."),
    ("human", "Topic: {topic}")
])

# Use `|` to chain the prompt and the structured model
standard_lcel_chain = prompt | model.with_structured_output(TopicReport)


# =====================================================================
# METHOD 2: The Self-Correcting LCEL Way (Using RunnableLambda)
# =====================================================================
# Since self-correction requires a stateful loop, there is no built-in `|` component. 
# Instead, we wrap the self-correcting logic in a `RunnableLambda`, which allows
# it to be chained using the `|` operator like any other runnable.

def self_correcting_loop(inputs: dict) -> TopicReport:
    topic = inputs["topic"]
    max_retries = inputs.get("max_retries", 3)
    
    # Compile prompt value
    prompt_val = prompt.invoke({"topic": topic})
    structured_model = model.with_structured_output(TopicReport)
    
    for attempt in range(max_retries):
        try:
            return structured_model.invoke(prompt_val)
        except (ValidationError, Exception) as e:
            if attempt == max_retries - 1:
                raise e
            print(f"\n[Attempt {attempt + 1} Failed]: {e}")
            print("Requesting the model to correct its output...")
            
            # Feed the validation error back to prompt messages so the LLM knows what to fix
            prompt_val.messages.extend([
                ("ai", "Invalid response structure or JSON schema violation."),
                ("human", f"Your previous response failed validation with the following error:\n{e}\n\nPlease correct your output and return it strictly according to the required schema.")
            ])

# Chain the self-correcting loop using the `|` operator
self_correcting_chain = RunnableLambda(self_correcting_loop)


if __name__ == "__main__":
    topic = "Indian premier league 2025/2026"
    
    # --- Running Method 1 ---
    print("--- Running Method 1 (Standard Modern LCEL) ---")
    result_method1 = standard_lcel_chain.invoke({"topic": topic})
    print(result_method1.model_dump_json(indent=2))
    
    # --- Running Method 2 ---
    print("\n--- Running Method 2 (Self-Correcting LCEL) ---")
    result_method2 = self_correcting_chain.invoke({"topic": topic, "max_retries": 3})
    print(result_method2.model_dump_json(indent=2))
