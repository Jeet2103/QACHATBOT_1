import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Initialize session state variables
if 'store' not in st.session_state:
    st.session_state.store = {}

if 'id' not in st.session_state:
    st.session_state.id = 0

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please give the best answer of all questions."),
        MessagesPlaceholder(variable_name="question"),
    ]
)

def increase_id():
    st.session_state.id += 1

def decrease_id():
    if st.session_state.id > 0:
        st.session_state.id -= 1

def generate_response(question, temp, max_tokens, llm="llama3-70b-8192"):
    config = {"configurable": {"session_id": f"Q{st.session_state.id}"}}
    model = ChatGroq(model=llm, groq_api_key=GROQ_API_KEY, temperature=temp, max_tokens=max_tokens)
    chain = prompt | model
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    response = with_message_history.invoke(
        {"question": [HumanMessage(content=question)]},
        config=config
    )
    return response.content

# Streamlit app UI
st.title("Simple Q&A ChatBot")

st.sidebar.title("Settings")

# Arrange buttons side by side
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button('New Chat'):
        increase_id()

with col2:
    if st.button("Previous Chat"):
        decrease_id()

llm = st.sidebar.selectbox("Select an Open source LLM Model:", ["llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it", "gemma-7b-it"])

temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Maximum Tokens", min_value=100, max_value=1000, value=300)

st.write("Go ahead and ask the question:")
question = st.text_input("Question:")

if question:
    response = generate_response(question, temp, max_tokens, llm)
    st.write(response)
else:
    st.write("Please enter a question")
