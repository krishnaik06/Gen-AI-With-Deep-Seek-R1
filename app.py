import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import uuid
from datetime import datetime

# Page config
st.set_page_config(
    page_title="DeepSeek Code Companion",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
        background-color: #2d2d2d !important;
        border-color: #4d4d4d !important;
    }
    
    /* Select box styling */
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
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* Chat history styling */
    .chat-history-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .chat-history-item:hover {
        background-color: #3d3d3d;
        border-color: #6d6d6d;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #4d4d4d;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #4d4d4d;
        border-color: #6d6d6d;
    }
    
    /* Chat container */
    .chat-container {
        border-radius: 10px;
        background-color: #2d2d2d;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Message styling */
    .user-message {
        background-color: #3d3d3d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .ai-message {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'chats' not in st.session_state:
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats = {
        new_chat_id: {
            'messages': [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': "New Chat"
        }
    }
    st.session_state.current_chat_id = new_chat_id

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Chat management functions
def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = {
        'messages': [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'title': "New Chat"
    }
    st.session_state.current_chat_id = new_chat_id
    st.session_state.awaiting_response = False

def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.awaiting_response = False

def get_chat_title(messages):
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
    return "New Chat"

# Sidebar layout
with st.sidebar:
    st.title("🧠 DeepSeek")
    
    # Configuration section
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-llm:latest", "huihui_ai/deepseek-r1-abliterated:8b"],
        index=0
    )
    
    st.divider()
    
    # Model capabilities section
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    
    st.divider()
    
    # Chat history section
    st.markdown("### 💬 Chat History")
    
    # New chat button
    if st.button("➕ New Chat", key="new_chat", use_container_width=True):
        create_new_chat()
    
    # Display chat history
    for chat_id, chat_data in sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]['timestamp'],
        reverse=True
    ):
        chat_title = get_chat_title(chat_data['messages'])
        
        # Create a clickable chat history item
        if st.button(
            f"💬 {chat_title}",
            key=f"chat_{chat_id}",
            help=f"Created: {chat_data['timestamp']}",
            use_container_width=True
        ):
            switch_chat(chat_id)
    
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Main chat interface
st.title("DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Initialize LLM engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert AI coding assistant called DeepSeek. Your specialties include:
- Writing clean, efficient code with best practices
- Debugging complex issues with strategic print statements
- Providing clear code documentation
- Designing elegant solutions to programming problems

Always provide concise, correct solutions and explain your reasoning clearly.
Respond in English and format code blocks appropriately using markdown.
""")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.chats[st.session_state.current_chat_id]['messages']:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.chats[st.session_state.current_chat_id]['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show processing message if waiting for response
    if st.session_state.awaiting_response:
        with st.chat_message("ai"):
            with st.spinner("🧠 Processing..."):
                prompt_chain = build_prompt_chain()
                ai_response = generate_ai_response(prompt_chain)
                st.session_state.chats[st.session_state.current_chat_id]['messages'].append(
                    {"role": "ai", "content": ai_response}
                )
                st.session_state.awaiting_response = False
                st.rerun()

# Chat input
user_query = st.chat_input("Type your coding question here...")

if user_query:
    # Add user message to log
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(
        {"role": "user", "content": user_query}
    )
    st.session_state.awaiting_response = True
    st.rerun()