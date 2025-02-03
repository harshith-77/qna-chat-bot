import os
import streamlit as st
import logging
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info("Initializing application...")

load_dotenv()

try:
    embeddings = HuggingFaceEmbeddings()
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=os.environ['GEMINI_API_KEY'], temperature=0)
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    logging.info("Models initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing LLM or embeddings: {e}")
    st.error("Failed to initialize AI components. Check logs for details.")
    st.stop()

st.title("Q&A ChatBot")

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "csv_name" not in st.session_state:
    st.session_state.csv_name = None

st.sidebar.header("Upload CSV for training:")
csv_file = st.sidebar.file_uploader(label='Upload CSV for training', type='.csv', label_visibility='collapsed')

if csv_file:
    if not os.path.exists("./uploaded_files"):
        os.makedirs("./uploaded_files")
    csv_save_path = f"./uploaded_files/{csv_file.name}"
    if st.session_state.csv_name != csv_file.name:
        try:
            with open(csv_save_path, 'wb') as f:
                f.write(csv_file.getbuffer())
            logging.info(f"CSV file saved: {csv_save_path}")

            csv_loader = CSVLoader(file_path=csv_save_path, autodetect_encoding=True)
            data = csv_loader.load()
            st.session_state.vectordb = FAISS.from_documents(data, embeddings)
            st.session_state.csv_name = csv_file.name
            logging.info("Vector database created successfully.")
            st.sidebar.success("Vector Database Created Successfully!!!")
        except Exception as e:
            logging.error(f"Error processing CSV file: {e}")
            st.sidebar.error("Failed to process CSV file. Check logs for details.")
    else:
        logging.info("The CSV File is already uploaded and processed")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt_template = '''
You are a helpful AI assistant. You are given the relevant context about the question. Generate an answer to the question based on the provided context only.
Do not make up answers. If you cannot answer the question based on the given context just say "I don't know". Answer only if you can give the answer based on the context provided.

CONTEXT = {context}

QUESTION = {question}
'''

query = st.chat_input("Enter your query")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    logging.info(f"Received user query: {query}")

    if st.session_state.vectordb is not None:
        try:
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k":20})
            compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
            reranker_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type='stuff',
                                                chain_type_kwargs={"prompt": prompt},
                                                retriever=reranker_retriever,
                                                return_source_documents=True)
            result = chain.invoke(query)
            logging.info(f"Query processed successfully. Result:\n{result}")
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            result = "I'm sorry, an error occurred while processing your request."

        st.chat_message("assistant").markdown(result["result"])
        st.session_state.messages.append({"role": "assistant", "content": result["result"]})

    else:
        st.sidebar.error("No CSV file uploaded. Please upload a file to gain knowledge first!!!.")
        logging.warning("Query attempted without an uploaded CSV.")