import streamlit as st
import time
from dataclasses import dataclass
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 10
# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")


# Initiate the chat engine with proper formatting
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template('''\
You are an expert AI coding assistant. Provide:
- Concise, correct solutions
- Strategic print statements for debugging
- Clear code explanations

Always respond in English and follow PEP8 guidelines.\
''')

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        ChatMessage("ai", "Hi! I'm DeepSeek. How can I help you code today? 💻", time.time())
    ]
if "last_request" not in st.session_state:
    st.session_state.last_request = 0

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message.role):
            st.markdown(message.content)

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain: ChatPromptTemplate) -> str:
    """Generate AI response with error handling."""
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        return f"Error generating response: {str(e)}"

def build_prompt_chain() -> ChatPromptTemplate:
    """Build conversation-aware prompt chain with type hints."""
    prompt_sequence = [system_prompt] + [
        HumanMessagePromptTemplate.from_template(msg.content) 
        if msg.role == "user" 
        else AIMessagePromptTemplate.from_template(msg.content)
        for msg in st.session_state.message_log
    ]
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Input validation
    if not user_query.strip():
        st.error("Please enter a valid question")
        st.stop()
        
    if len(user_query) > 1000:
        st.error("Question too long (max 1000 characters)")
        st.stop()

    # Rate limiting
    current_time = time.time()
    if current_time - st.session_state.last_request < 60/MAX_REQUESTS_PER_MINUTE:
        st.error("Too many requests. Please wait a moment.")
        st.stop()
    
    st.session_state.last_request = current_time
    st.session_state.message_log.append(ChatMessage("user", user_query, current_time))
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append(ChatMessage("ai", ai_response, time.time()))
    
    # Rerun to update chat display
    st.rerun()
