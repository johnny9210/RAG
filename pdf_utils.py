# pdf_utils.py

import io
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pdf_analyzer import GraphState, analyze_pdf, extract_tag_elements_per_page, extract_page_elements, create_image_summary, create_text_summary
from pdf_analyzer import split_pdf, analyze_layout, extract_page_metadata
import tempfile
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 경로를 얻습니다 (현재 디렉토리의 부모 디렉토리).
project_root = os.path.dirname(current_dir)
# .env 파일의 경로를 지정합니다.
dotenv_path = os.path.join(project_root, '.env')

# .env 파일이 존재하는지 확인합니다.
if os.path.exists(dotenv_path):
    print(f".env file found at {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f".env file not found at {dotenv_path}")
    sys.exit(1)  # .env 파일이 없으면 프로그램을 종료합니다.

es_cloud_id = os.getenv("ELASTICSEARCH_CLOUD_ID")
es_username = os.getenv("ELASTICSEARCH_USERNAME")
es_password = os.getenv("ELASTICSEARCH_PASSWORD")

if all([es_cloud_id, es_username, es_password]):
    try:
        es_client = Elasticsearch(
            cloud_id=es_cloud_id,
            basic_auth=(es_username, es_password)
        )
        # Elasticsearch 연결 테스트
        if es_client.ping():
            print("Successfully connected to Elasticsearch")
            es_url = f"https://{es_cloud_id.split(':')[1]}"
        else:
            print("Failed to connect to Elasticsearch")
            es_client = None
            es_url = None
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        es_client = None
        es_url = None
else:
    print("Elasticsearch connection info is incomplete. Elasticsearch features will be disabled.")
    es_client = None
    es_url = None



if es_cloud_id:
    es_client = Elasticsearch(
        cloud_id=es_cloud_id,
        basic_auth=(es_username, es_password)
    )
else:
    es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    es_client = Elasticsearch(
        hosts=[es_host],
        basic_auth=(es_username, es_password)
    )

def create_elasticsearch_store(text, index_name="pdf_content"):
    if es_client is None:
        print("Elasticsearch client is not initialized. Vector store creation failed.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    
    vector_store = ElasticsearchStore(
        index_name=index_name,
        embedding=embeddings,
        es_cloud_id=es_cloud_id,
        es_user=es_username,
        es_password=es_password
    )
    
    vector_store.add_texts(texts)
    return vector_store 

def process_pdf(uploaded_file):
    try:
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF 처리 중 오류 발생: {str(e)}")
        return None


# def create_vector_store(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(text)
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(texts, embeddings)
#     return vector_store


def handle_pdf_upload(uploaded_files):
    pdf_text = ""
    for uploaded_file in uploaded_files:
        current_text = process_pdf(uploaded_file)
        if current_text:
            pdf_text += current_text + "\n\n"
    if pdf_text:
        vector_store = create_elasticsearch_store(pdf_text)
        return pdf_text, vector_store
    return None, None


def get_answer_from_elasticsearch(question, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables or Streamlit secrets")
    
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o",
        openai_api_key=api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": question})
    return result['result'], result['source_documents']


def analyze_pdf(uploaded_file, batch_size: int = 10):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    state = GraphState(filepath=temp_file_path, batch_size=batch_size)
    state.update(split_pdf(state))
    state.update(analyze_layout(state))
    state.update(extract_page_metadata(state))
    state.update(extract_page_elements(state))
    state.update(extract_tag_elements_per_page(state))
    
    # page_numbers를 명시적으로 추가
    state['page_numbers'] = list(state['page_elements'].keys())
    
    state.update(extract_page_text(state))
    state.update(create_text_summary(state))
    state.update(create_image_summary(state))

    os.unlink(temp_file_path)

    return state

def extract_page_text(state: GraphState):
    page_numbers = state.get("page_numbers", [])
    if not page_numbers:
        # page_numbers가 없으면 page_elements의 키를 사용
        page_numbers = list(state.get("page_elements", {}).keys())
    
    extracted_texts = {}
    for page_num in page_numbers:
        extracted_texts[page_num] = ""
        for element in state.get("page_elements", {}).get(page_num, {}).get("text_elements", []):
            extracted_texts[page_num] += element.get("text", "")
    return {"texts": extracted_texts}

def get_answer_from_pdf(question, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables or Streamlit secrets")
    
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o",  # 'model_name' 대신 'model' 사용
        openai_api_key=api_key  # 'api_key' 대신 'openai_api_key' 사용
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": question})
    return result['result'], result['source_documents']

def display_pdf_upload_section():
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        pdf_text, vector_store = handle_pdf_upload(uploaded_files)
        if pdf_text and vector_store:
            st.session_state['pdf_text'] = pdf_text
            st.session_state['vector_store'] = vector_store
            st.success("PDF 파일들이 성공적으로 업로드되고 분석되었습니다.")
        return uploaded_files
    return None