from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

def generate_content(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, api_key=api_key)
    chat_prompt = ChatPromptTemplate.from_template(prompt)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain.run(input=prompt)




