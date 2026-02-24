import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import pickle
warnings.filterwarnings("ignore")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai  import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")
file_path = "faiss_store_openai.pkl"

 
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7)

process_url_clicked=st.sidebar.button("Process URLs")

main_window=st.empty()

if process_url_clicked:
    main_window.text("Processing URLs...")
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=1000, chunk_overlap=200)
    main_window.text("Splitting Started...")
    texts = text_splitter.split_documents(data)
    encoder=OpenAIEmbeddings()
    main_window.text("Encoding Started...")
    vectorstore_openai=FAISS.from_documents(texts,encoder)
    main_window.text("Vector Store Created!")
    vectorstore_openai.save_local("faiss_index")
    main_window.text("Vector Store Saved!")

