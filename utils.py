#utils.py
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import yaml
import os
from langchain.vectorstores import FAISS
from pdf_utils import handle_pdf_upload, process_pdf, create_elasticsearch_store
import json
from mongodb_utils import connect_to_mongodb, get_template, get_all_template_names, get_templates_from_files
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from pdf_analyzer import analyze_pdf, generate_toc_from_description
from content_generator import generate_content



load_dotenv()


@st.cache_resource
def get_openai_model():
    return ChatOpenAI(temperature=0.7, model_name="gpt-4o")

def load_template(template_name, path):
    json_path = os.path.join(path, f"{template_name}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            template_data = json.load(file)
        
        return {
            "name": template_name,
            "introduction": f"{template_name} 템플릿 봇입니다.",
            "personality": template_data.get('personality', "친절하고 도움이 되는 성격"),
            "image": template_data.get('image'),
            "template": template_data.get('template', ''),
            "input_variables": template_data.get('input_variables', [])
        }
    else:
        st.error(f"Template {template_name} not found.")
    return None

def get_basic_templates(path):
    templates = []
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            templates.append(os.path.splitext(filename)[0])
    return templates

def generate_toc(session_state):
    return generate_toc_from_description(session_state, generate_content)

def generate_content_for_section(title, service):
    prompt = f"""
    서비스: {service}
    섹션 제목: {title}
    
    위 정보를 바탕으로 해당 섹션의 내용을 생성해주세요. 
    내용은 구체적이고 관련성 있어야 하며, 최소 200단어 이상으로 작성해 주세요.
    """
    return generate_content(prompt)

def chat_with_bot(bot_info):
    st.title(f"Chat with {bot_info['name']}")
    st.write(bot_info['introduction'])
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    
    if 'pdf_text' not in st.session_state:
        st.session_state['pdf_text'] = None
    
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    
    # PDF upload section
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        pdf_texts = []
        for uploaded_file in uploaded_files:
            pdf_text = analyze_pdf(uploaded_file)
            if pdf_text:
                pdf_texts.append(pdf_text)
        
        if pdf_texts:
            st.session_state['pdf_text'] = "\n\n".join(pdf_texts)
            st.session_state['vector_store'] = create_elasticsearch_store(st.session_state['pdf_text'])
            st.success("PDF 파일들이 성공적으로 업로드되고 분석되었습니다.")
    
    for message in st.session_state['messages']:
        st.chat_message(message["role"]).markdown(message["content"])
    
    user_question = st.text_area("분석을 시작하려면 '분석 시작'이라고 입력하세요:", key="user_question")
    
    if st.button("분석 시작"):
        if not user_question or user_question.lower() != "분석 시작":
            st.warning("'분석 시작'이라고 입력해주세요.")
            return

        if not st.session_state['pdf_text']:
            st.warning("먼저 PDF 파일을 업로드해주세요.")
            return

        st.session_state['messages'].append({"role": "user", "content": user_question})
        st.chat_message("user").markdown(user_question)
        
        try:
            prompt_template = st.session_state["prompt_template"]
            chain = st.session_state["chain"]
            
            # YAML 템플릿의 input_variables에 따라 동적으로 입력 변수 설정
            input_variables = prompt_template.input_variables
            inputs = {
                "question": "학생들의 과제를 분석해주세요.",
                "context": st.session_state['pdf_text']
            }
            for var in input_variables:
                if var not in inputs and var in st.session_state:
                    inputs[var] = st.session_state[var]
            
            response = chain.run(**inputs)
            
            st.session_state['messages'].append({"role": "assistant", "content": response})
            st.chat_message("assistant").markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Chain configuration:", st.session_state["chain"])
    
    if st.button("대화기록 초기화"):
        st.session_state['messages'] = []
        st.session_state['pdf_text'] = None
        st.session_state['vector_store'] = None
        st.experimental_rerun()

def update_bot_index(bots):
    embeddings = OpenAIEmbeddings()
    texts = [f"{bot['name']} {' '.join(bot['hashtags'])}" for bot in bots]
    metadatas = [{'name': bot['name'], 'hashtags': bot['hashtags']} for bot in bots]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)