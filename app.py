import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
load_dotenv()

#Langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q/A Chat With Opensource LLM Models"


#creating prompt
prompt =  ChatPromptTemplate.from_messages([
    ("system","Your are helpful assistant. Please respond to the user queries accurately"),
    ("user","Question:{Question}")
])

def generate_response(question,engine,temperature,max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"Question":question})
    return answer

#Streamlit app
st.title("Enhanced Q/A Chatbot With Multiple Opensource LLM Models")

#Select LLM model
engine = st.sidebar.selectbox("Select Opensource LLM Model",["gemma2:2b","mistral"])

#Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

#Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provider user input")




