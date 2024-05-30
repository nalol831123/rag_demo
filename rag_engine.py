import os, json
from pathlib import Path
import sqlite3
from datetime import datetime

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_feedback import streamlit_feedback
from langchain.prompts import PromptTemplate

import streamlit as st

#TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    st.session_state.db_name = st.text_input("Database name")

def load_documents(TMP_DIR):
    documents = []
    for file in os.listdir(TMP_DIR.as_posix()):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(TMP_DIR.as_posix(), file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs)
    
    LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store', str(st.session_state.db_name))

    vectordb = Chroma.from_documents(texts, embedding=embeddings,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    del st.session_state['source_docs']
    del st.session_state['db_name']

    return "File uploaded successfully! Please switch to Chat tab"

def process_documents():
    try:
        UPLOAD_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp', str(st.session_state.db_name))
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        for source_doc in st.session_state.source_docs:
            filename = source_doc.name
            with open(os.path.join(UPLOAD_DIR, filename), "wb") as tmp_file:
                tmp_file.write(source_doc.read())

        documents = load_documents(UPLOAD_DIR)   
        texts = split_documents(documents)
        st.success(embeddings_on_local_vectordb(texts))

    except Exception as e:
        st.error(f"An error occurred: {e}")

def define_llm():
    # llm = ChatOpenAI(openai_api_key=st.session_state.key, openai_api_base=st.session_state.base)
    llm = ChatOpenAI(openai_api_key=st.session_state.key, openai_api_base=st.session_state.base)
    return llm

def vectordb(model):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    # model_kwarg = {}
    # if st.session_state.base == 'llama':
    #     model_kwarg = {'device': 'cuda:0'}
    # elif st.session_state.base == 'taide':
    #     model_kwarg = {'device': 'cuda:1'}
    # embedding = HuggingFaceEmbeddings(model_name=model_name,
    #                                 model_kwargs=model_kwarg)
    embedding = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs)
    path = Path(__file__).resolve().parent.joinpath('data', 'vector_store', st.session_state.db_name)
    vectordb = Chroma(persist_directory=path.as_posix(), embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={'k': 2})
    return retriever

def query_llm(query, model):
    if model == 'taide':
        template = """給以下歷史對話與後續問題，將後續問題改寫為獨立問題

        歷史對話:
        {chat_history}
        後續問題: {question}
        獨立問題: """

        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )

        custom_prompt_template = """使用下面的內容來回答以下的問題. 如果你不知道答案就說你不知道, 不要試圖編造答案.

            {context}

            問題: {question}
            請用中文回答:"""

        # Create a PromptTemplate instance with your custom template
        custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"],
        )

        # Use your custom prompt when creating the ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm = define_llm(),
            retriever = vectordb(model),
            return_source_documents = True,
            condense_question_prompt=prompt,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
        )
        result = qa_chain({'question': query, 'chat_history': st.session_state.messages})

    else:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm = define_llm(),
            retriever = vectordb(model),
            return_source_documents = True,
        )
        result = qa_chain({'question': query, 'chat_history': st.session_state.messages})


    st.session_state.messages.append((query, result['answer'], result['source_documents']))
    return result['answer'], result['source_documents']

def list_folders():
    path = Path(__file__).resolve().parent.joinpath('data', 'vector_store')
    folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return folder_names

def Create_db():
    st.title("Upload file to create db for RAG")
    # upload 
    input_fields()
    st.button("Submit Documents", on_click=process_documents)

def read_model():
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)
    return model_config

def create_dialog_db():
    # Initialize SQLite database connection
    conn = sqlite3.connect('dialog.db')
    c = conn.cursor()

    # Create table to store dialog messages if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS dialog
                (model TEXT, rag_database TEXT, user_iput TEXT, model_output TEXT, eval TEXT, comment TEXT, timestamp TEXT)''')
    return conn, c

def _submit_feedback(user_response, human, ai):
    st.toast(f"Feedback submitted!!!! Thank you!!!!")
    # create dialog db
    conn, c = create_dialog_db()
    c.execute("INSERT INTO dialog (model, rag_database, user_iput, model_output, eval, comment, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", (st.session_state.model, st.session_state.db_name, human, ai,  map_score(user_response['score']), user_response['text'], datetime.now()))
    conn.commit()

def map_score(score):
    return 1 if score == '👍' else 0

def Chat():
    st.title("RAG")
    # sidebar
    model_config = read_model()
    st.session_state.model = st.sidebar.selectbox("Select model", list(model_config.keys()))
    st.session_state.base = model_config[st.session_state.model]
    st.session_state.key = 'None'

    # select db
    st.session_state.db_name = st.sidebar.selectbox("Select RAG Database", list_folders())

    # chat
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #  print messages
    n = 0
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
        st.chat_message('ai').write(message[2])  
        # for feedback
        feedback_key = f"feedback_{int(n)}"
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None

        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="Please provide extra information",
            on_submit = _submit_feedback,
            key=feedback_key,
            kwargs={
                "human": message[0],
                "ai": message[1],
            },
        )
        n += 1

    # input
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response, cite = query_llm(query, st.session_state.model)
        message = st.chat_message("ai")
        message.write(response)
        message.write("Citation:")
        message.write(cite)
        st.rerun()

# title
st.set_page_config(page_title="RAG")
 
# sidebar
selected_tab = st.sidebar.selectbox("Select Function", ["Chat", "Create Database"])
if selected_tab == "Chat":
    Chat()
elif selected_tab == "Create Database":
    Create_db()