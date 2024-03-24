import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, ImageCaptionLoader
import os
import tempfile
from transformers.utils import logging

logging.set_verbosity_error()

class TextDocument:
    def __init__(self, content):
        self.page_content = content
        self.metadata = ""  
        
        


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, ask me anything!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hiiii!"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me anything", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                reply_container.chat_message("user", avatar="💀").write(st.session_state["past"][i])
                reply_container.chat_message("assistant", avatar="🤖").write(st.session_state["generated"][i])

def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
    streaming = True,
    model_path="./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    temperature=0.75,
    top_p=1, 
    verbose=True,
    n_ctx=4096
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                memory=memory)
    return chain


def main():
    # Initialize session state
    initialize_session_state()

    st.title("Class Skippers' Comprehension Improver (CSCI)🎉🎉🎉")

    # Initialize Streamlit
    st.sidebar.title("Files Processing:")
    uploaded_files_pdf = st.sidebar.file_uploader("Upload PDFs 📋:", accept_multiple_files=True,type=["pdf", "doc", "docx", "txt"], key="pdf")
    uploaded_files_image = st.sidebar.file_uploader("Upload Images 📸:", accept_multiple_files=True, type=["jpg", "jpg", "png"], key="image")
    uploaded_files_recording = st.sidebar.file_uploader("Upload Recordings 📹: (In developing process)", accept_multiple_files=True, key="recording")
    
    if uploaded_files_pdf or uploaded_files_image or uploaded_files_recording:
        text = []
        for file in uploaded_files_pdf:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        for file in uploaded_files_image:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            
            loader = ImageCaptionLoader(temp_file_path).load()
            if loader:
                text.extend(loader)
                os.remove(temp_file_path)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})  
        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)
        
    
    

    
if __name__ == "__main__":
    main()