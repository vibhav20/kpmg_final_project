import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Constants for environment setup
GOOGLE_API_KEY = "AIzaSyBm3UYleYvRMoFzTiFJzucfVjnvTx38mS8"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Function definitions
def extract_pdf_contents(filenames):
    combined_text = ""
    for file in filenames:
        reader = PdfReader(file)
        for page in reader.pages:
            combined_text += page.extract_text() or ""
    return combined_text

def segment_text(text, size=1000, overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings_generator = GooglePalmEmbeddings()
    store = FAISS.from_texts(chunks, embedding=embeddings_generator)
    return store

def create_chat_chain(store):
    model = GooglePalm()
    convo_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=model, retriever=store.as_retriever(), memory=convo_memory
    )
    return chat_chain

def handle_user_query(question):
    chat_response = st.session_state.chat_chain({"question": question})
    st.session_state.chat_history = chat_response["chat_history"]
    for idx, msg in enumerate(st.session_state.chat_history):
        speaker = "You" if idx % 2 == 0 else "Assistant"
        st.write(f"{speaker}: {msg.content}")

def main():
    st.set_page_config(page_title="PDF Interaction Hub")
    st.header("PDF Query Tool🗨️")

    query = st.text_input("Pose a Question Based on the PDF")
    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if query and st.session_state.chat_chain:
        handle_user_query(query)

    with st.sidebar:
        st.title("Configuration")
        st.subheader("PDF Upload")
        pdfs = st.file_uploader(
            "Choose PDF file, then Process",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process PDF"):
            with st.spinner("Extracting & Processing..."):
                text_from_pdfs = extract_pdf_contents(pdfs)
                text_segments = segment_text(text_from_pdfs)
                vectors = build_vector_store(text_segments)
                st.session_state.chat_chain = create_chat_chain(vectors)
                st.success("Setup Complete")

if __name__ == "__main__":
    main()
