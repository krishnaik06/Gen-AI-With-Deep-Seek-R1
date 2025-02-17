import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time

# Initialize session state variables
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "deepseek-r1:1.5b"  # Default to smaller model
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model=st.session_state.selected_model))
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    
    /* Think Section Styling */
    .think-section {
        color: #666666 !important;
        font-style: italic !important;
        font-size: 0.9em !important;
        border-left: 3px solid #444444 !important;
        padding-left: 10px !important;
        margin: 10px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_size = st.selectbox(
        "Select Model Size",
        ["deepseek-r1:1.5b", "deepseek-r1:7b"],
        help="Smaller model (1.5b) is faster but less accurate. Larger model (7b) is slower but more accurate.",
        index=0
    )
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=800,
        step=100,
        help="Smaller chunks process faster but may miss context. Larger chunks are slower but maintain more context."
    )
    
    if model_size != st.session_state.selected_model:
        st.session_state.selected_model = model_size
        st.session_state.document_loaded = False
        st.rerun()

# Update constants with selected configuration
EMBEDDING_MODEL = OllamaEmbeddings(model=st.session_state.selected_model)
LANGUAGE_MODEL = OllamaLLM(model=st.session_state.selected_model)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
First, think through the problem (wrap your thinking in <think></think> tags).
Then provide your final answer (max 3 sentences).
If unsure, state that you don't know.

Query: {user_query}
Context: {document_context}
Response:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'


def save_uploaded_file(uploaded_file):
    os.makedirs("document_store/pdfs", exist_ok=True)

    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Update chunk_documents function with configurable chunk size
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 4,  # Overlap of 25%
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Update the index_documents function
def index_documents(document_chunks):
    st.session_state.vector_store.add_documents(document_chunks)

# Update the find_related_documents function
def find_related_documents(query):
    return st.session_state.vector_store.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    
    # Format the response with styled think section
    if "<think>" in response and "</think>" in response:
        parts = response.split("</think>", 1)
        think_part = parts[0].replace("<think>", "").strip()
        answer_part = parts[1].strip() if len(parts) > 1 else ""
        formatted_response = f"""<div class="think-section">{think_part}</div>
        
{answer_part}"""
        return formatted_response
    return response


# UI Configuration


st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant which runs locally in your browser!")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf and not st.session_state.document_loaded:
    try:
        progress_bar = st.progress(0, text="Starting document processing...")
        status_container = st.empty()
        
        # Step 1: Save file
        progress_bar.progress(0.1, text="Saving uploaded file...")
        saved_path = save_uploaded_file(uploaded_pdf)
        time.sleep(0.5)  # Small delay for UI feedback
        
        # Step 2: Load PDF
        progress_bar.progress(0.2, text="Loading PDF document...")
        status_container.info("📄 Extracting text from PDF...")
        raw_docs = load_pdf_documents(saved_path)
        if not raw_docs:
            progress_bar.empty()
            status_container.empty()
            st.error("No text could be extracted from the PDF.")
            st.stop()
        
        # Step 3: Process chunks
        progress_bar.progress(0.4, text="Processing document chunks...")
        status_container.info(f"📑 Processing {len(raw_docs)} pages into chunks...")
        processed_chunks = chunk_documents(raw_docs)
        if not processed_chunks:
            progress_bar.empty()
            status_container.empty()
            st.error("Could not process the document into chunks.")
            st.stop()
        
        # Step 4: Index documents
        progress_bar.progress(0.6, text="Creating document index...")
        status_container.info(f"🔍 Indexing {len(processed_chunks)} text chunks...")
        index_documents(processed_chunks)
        
        # Step 5: Finalizing
        progress_bar.progress(1.0, text="Completed!")
        status_container.success(f"✅ Successfully processed {len(raw_docs)} pages into {len(processed_chunks)} searchable chunks!")
        time.sleep(1)  # Show completion message briefly
        
        # Clean up UI elements
        progress_bar.empty()
        status_container.empty()
        
        st.session_state.document_loaded = True
        st.success("✅ Document processed successfully! Ask your questions below.")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_container.empty()
        st.error(f"Error processing document: {str(e)}")
        st.session_state.document_loaded = False
        st.stop()

# Update the chat interface section
if st.session_state.document_loaded:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        try:
            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                if not relevant_docs:
                    ai_response = "I couldn't find any relevant information in the document for your question."
                else:
                    ai_response = generate_answer(user_input, relevant_docs)
                
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Display AI response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(ai_response, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Update the reset button to also clear chat history
if st.session_state.document_loaded:
    if st.button("Reset Document"):
        st.session_state.document_loaded = False
        st.session_state.vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model=st.session_state.selected_model))
        st.session_state.chat_history = []  # Clear chat history
        st.rerun()
