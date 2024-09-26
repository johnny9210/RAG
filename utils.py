#utils.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import yaml
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pdf_analyzer import (
    process_pdf, 
    create_vector_store, 
    answer_question, 
    generate_toc_from_description
)
from langchain.chains import LLMChain, RetrievalQA


TEMPLATE_DIR = "/Users/ilgyun/Documents/langchain/langchain-kr-main/19-Streamlit/03-test/prompts"

@st.cache_resource
def get_openai_model():
    return ChatOpenAI(temperature=0.7, model_name="gpt-4o")

def generate_content(prompt):
    llm = get_openai_model()
    chat_prompt = ChatPromptTemplate.from_template(prompt)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain.run(input=prompt)

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

def load_template(path, template_name):
    base_path = path
    yaml_files = os.listdir(base_path)
    template_file = next((f for f in yaml_files if f.lower().replace(" ", "") == f"{template_name.lower()}.yaml"), None)
    if template_file:
        yaml_path = os.path.join(base_path, template_file)
        try:
            with open(yaml_path, 'r', encoding='utf8') as file:
                yaml_content = yaml.safe_load(file)
            if isinstance(yaml_content, dict):
                template = yaml_content.get('template', '')
                input_variables = yaml_content.get('input_variables', ['question'])
                prompt = ChatPromptTemplate.from_template(template)
                st.session_state["prompt_template"] = prompt
                st.session_state["chain"] = LLMChain(llm=get_openai_model(), prompt=prompt)
                return {
                    "name": template_name,
                    "introduction": f"{template_name} 템플릿 봇입니다.",
                    "personality": "친절하고 도움이 되는 성격",
                    "image": None,
                    "input_variables": input_variables
                }
            else:
                raise ValueError("Unexpected YAML content format")
        except Exception as e:
            st.error(f"Error loading YAML file: {str(e)}")
    else:
        st.error(f"YAML file for {template_name} not found.")
    return None

def get_basic_templates(path):
    yaml_files = [f for f in os.listdir(path) if f.endswith('.yaml')]
    return [os.path.splitext(f)[0] for f in yaml_files]


# def chat_with_bot(bot_info):
#     st.title(f"Chat with {bot_info['name']}")
#     st.write(bot_info['introduction'])
    
#     if 'messages' not in st.session_state:
#         st.session_state['messages'] = []
    
#     if 'pdf_text' not in st.session_state:
#         st.session_state['pdf_text'] = None
    
#     if 'vector_store' not in st.session_state:
#         st.session_state['vector_store'] = None
    
#     # PDF upload section (exclude for 프롬프트_메이커)
#     if bot_info['name'] != "프롬프트_메이커":
#         uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
#         if uploaded_file:
#             pdf_text = process_pdf(uploaded_file)
#             if pdf_text:
#                 st.session_state['pdf_text'] = pdf_text
#                 st.session_state['vector_store'] = create_vector_store(pdf_text)
#                 st.success("PDF가 성공적으로 업로드되고 분석되었습니다.")
    
#     for message in st.session_state['messages']:
#         st.chat_message(message["role"]).markdown(message["content"])
    
#     user_question = st.text_area("Enter your question:", key="user_question")
    
#     if st.button("답변 생성"):
#         if not user_question:
#             st.warning("질문을 입력해주세요.")
#             return

#         st.session_state['messages'].append({"role": "user", "content": user_question})
#         st.chat_message("user").markdown(user_question)
        
#         try:
#             if bot_info['name'] != "프롬프트_메이커" and st.session_state['vector_store']:
#                 context = st.session_state['pdf_text']
#                 chain = st.session_state["chain"]
#                 response = chain.invoke({
#                     "question": user_question,
#                     "context": context
#                 })
#             else:
#                 chain = st.session_state["chain"]
#                 response = chain.invoke({"question": user_question})
            
#             formatted_response = response['text'] if isinstance(response, dict) else response
            
#             st.session_state['messages'].append({"role": "assistant", "content": formatted_response})
#             st.chat_message("assistant").markdown(formatted_response)
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#             st.write("Chain configuration:", st.session_state["chain"])
    
#     if st.button("대화기록 초기화"):
#         st.session_state['messages'] = []
#         st.experimental_rerun()   

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
            pdf_text = process_pdf(uploaded_file)
            if pdf_text:
                pdf_texts.append(pdf_text)
        
        if pdf_texts:
            st.session_state['pdf_text'] = "\n\n".join(pdf_texts)
            st.session_state['vector_store'] = create_vector_store(st.session_state['pdf_text'])
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

def handle_pdf_upload(uploaded_files):
    pdf_text = ""
    for uploaded_file in uploaded_files:
        pdf_text += process_pdf(uploaded_file) + "\n\n"
    if pdf_text:
        vector_store = create_vector_store(pdf_text)
        return pdf_text, vector_store
    return None, None

def get_answer_from_pdf(question, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model_name='gpt-4o')
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": question})
    return result['result'], result['source_documents']
