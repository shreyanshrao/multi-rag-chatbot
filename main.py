import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List
import time
import torch

#load env variables
load_dotenv()

working_dir=os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    """Load PDF using PyPDFLoader - much faster than UnstructuredPDFLoader"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_multiple_documents(file_paths: List[str]):
    """Load and combine documents from multiple PDF files"""
    all_documents = []
    for file_path in file_paths:
        documents = load_document(file_path)
        all_documents.extend(documents)
    return all_documents

def get_embeddings():
    """Get or create cached embeddings model - using fast, lightweight model"""
    if "embeddings" not in st.session_state:
        # Automatically detect if CUDA is available, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Using a fast, lightweight model optimized for speed
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, lightweight model
            model_kwargs={'device': device},  # Auto-detect device (CUDA if available, else CPU)
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
        
        # Display device info in sidebar
        if device == 'cuda':
            st.sidebar.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.info("💻 Using CPU for embeddings")
    return st.session_state.embeddings

def setup_vectorstore(documents):
    """Setup vectorstore with optimized settings for speed"""
    embeddings = get_embeddings()
    # Optimize chunking - larger chunks = fewer embeddings needed
    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # Split on paragraphs for better context
        chunk_size=1500,  # Slightly larger chunks = fewer embeddings
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or "gemini-1.5-flash" for faster/cheaper
        temperature=0,
    )
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks for better context
    )
    memory=ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=False
    )
    return chain
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)
st.title("🦙 Chat with your Document")

#initializing the chat history

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

uploaded_files=st.file_uploader(
    label="Upload your PDFs (up to 3 files)", 
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    # Limit to 3 files
    if len(uploaded_files) > 3:
        st.warning("⚠️ Maximum 3 PDFs allowed. Only the first 3 files will be processed.")
        uploaded_files = uploaded_files[:3]
    
    # Save uploaded files and collect paths
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = f"{working_dir}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    # Display uploaded files
    st.success(f"✅ Successfully uploaded {len(file_paths)} PDF file(s):")
    for i, file_path in enumerate(file_paths, 1):
        st.write(f"{i}. {os.path.basename(file_path)}")
    
    # Process documents if vectorstore doesn't exist or files have changed
    if st.session_state.vectorstore is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        # Step 1: Load PDFs
        status_text.text("📄 Loading PDF files...")
        progress_bar.progress(10)
        all_documents = load_multiple_documents(file_paths)
        load_time = time.time() - start_time
        
        # Step 2: Create embeddings and vectorstore
        status_text.text("🔢 Creating embeddings and vector store...")
        progress_bar.progress(50)
        embed_start = time.time()
        st.session_state.vectorstore = setup_vectorstore(all_documents)
        embed_time = time.time() - embed_start
        
        # Step 3: Create conversation chain
        status_text.text("🔗 Setting up conversation chain...")
        progress_bar.progress(90)
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
        
        total_time = time.time() - start_time
        progress_bar.progress(100)
        
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Documents processed successfully in {total_time:.1f}s! (Load: {load_time:.1f}s, Embed: {embed_time:.1f}s)")
        st.info(f"📊 Processed {len(all_documents)} pages from {len(file_paths)} file(s)")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input - only enable if PDFs are uploaded
if st.session_state.conversation_chain is None:
    user_input = st.chat_input("Please upload PDFs first...", disabled=True)
else:
    user_input = st.chat_input("Ask your question...")

if user_input and st.session_state.conversation_chain:
    st.session_state.chat_history.append({"role":"user","content":user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation_chain({"question": user_input})
                assistant_response = response["answer"]
                st.markdown(assistant_response)
                st.session_state.chat_history.append({"role":"assistant","content":assistant_response})
            except Exception as e:
                error_message = f"Sorry, an error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append({"role":"assistant","content":error_message})

# Display session info
if st.session_state.conversation_chain:
    st.sidebar.info(f"💬 Questions in this session: {len([m for m in st.session_state.chat_history if m['role'] == 'user'])}")
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        # Reset memory in the chain
        if st.session_state.conversation_chain:
            st.session_state.conversation_chain.memory.clear()
        st.rerun()