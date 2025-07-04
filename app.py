import os
import sqlite3
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="Groq Chatbot", layout="wide")

# --- Initialize DB ---
def init_db():
    conn = sqlite3.connect("chatbot_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_query_to_db(question):
    conn = sqlite3.connect("chatbot_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_queries (question) VALUES (?)", (question,))
    conn.commit()
    conn.close()

init_db()  # Ensure DB is initialized once

# --- CSS Styling ---
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #111111;
        }

        .main-container {
            animation: slideIn 0.6s ease-out;
            background: transparent !important;
            color: white;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(40px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .history-item {
            padding: 10px;
            margin: 5px 0;
            background-color: transparent !important;
            border-radius: 8px;
            font-size: 15px;
            color: white;
            border-bottom: 1px solid #444;
        }

        .response-box {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            font-size: 16px;
            line-height: 1.6;
            color: white;
        }

        .title {
            text-align: center;
            font-size: 2em;
            font-weight: 600;
            color: white;
            margin-top: 20px;
        }

        .subtitle {
            text-align: center;
            font-size: 16px;
            color: #aaa;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# --- LangChain Setup ---
model = init_chat_model("groq:llama-3.1-8b-instant")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide response to the user queries."),
    ("user", "Question: {question}")
])
llm = ChatGroq(model="llama-3.1-8b-instant")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR: Chat History ---
with st.sidebar:
    st.markdown("### Chat History")
    st.markdown('<div class="sidebar-history">', unsafe_allow_html=True)
    for message in reversed(st.session_state.history):
        st.markdown(f"<div class='history-item'>{message}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Clear History"):
        st.session_state.history = []

# --- MAIN AREA ---
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("<div class='title'>Groq Chatbot with LLaMA 3</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ask questions. Get intelligent answers — fast.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        input_text = st.text_input("Enter your question:", label_visibility="collapsed", placeholder="e.g., What is LangChain?", key="user_input")
    with col2:
        submitted = st.button("Generate")

    if submitted and input_text.strip() != "":
        question = input_text.strip()

        # Save to sidebar + database
        st.session_state.history.append(question)
        save_query_to_db(question)

        # Generate response
        with st.spinner("Generating response..."):
            response = chain.invoke({'question': question})

        # Show response
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)