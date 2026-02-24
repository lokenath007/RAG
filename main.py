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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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



query = st.text_input("Ask a question about the articles:")
BASE_DIR = os.path.dirname(__file__)
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
if query:
    if not os.path.exists("faiss_index"):
        st.warning("Please process URLs first.")
    else:
        with st.spinner("Retrieving answer..."):

            embeddings = OpenAIEmbeddings()

            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

            # Prompt
            prompt = ChatPromptTemplate.from_template(
                """
                Answer the question based only on the context below:

                {context}

                Question: {question}
                """
            )

            # LCEL chain (LangChain v1 way)
            chain = (
                {
                    "context": retriever,
                    "question": lambda x: x
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(query)

        st.subheader("Answer")
        st.write(response)