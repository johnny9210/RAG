import streamlit as st
from dotenv import load_dotenv

from utils import get_basic_templates, load_template, update_bot_index, handle_pdf_upload, get_answer_from_pdf
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import math
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import HumanMessage
import uuid
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import pandas as pd
from langchain.schema import SystemMessage
import os

load_dotenv()

base_path = '/Users/ilgyun/Documents/langchain/langchain-kr-main/19-Streamlit/03-test/base_prompts'
basic_path = '/Users/ilgyun/Documents/langchain/langchain-kr-main/19-Streamlit/03-test/basic_prompts'

def chat_with_bot(bot_info):
    st.title(f"Chat with {bot_info['name']}")
    st.write(bot_info['introduction'])
    
    bot_key = f"bot_{bot_info['name']}"
    
    initialize_session_state(bot_key, bot_info)

    if bot_info['name'] == "기업애로사항 상담 봇":
        handle_enterprise_consultation_bot(bot_info, bot_key)
    else:
        handle_general_bot(bot_info, bot_key)

    if st.button("대화기록 초기화"):
        reset_chat_history(bot_key, bot_info)

def initialize_session_state(bot_key, bot_info):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = {}
    if bot_key not in st.session_state['messages']:
        st.session_state['messages'][bot_key] = []
    
    if 'initial_questions' not in st.session_state:
        st.session_state['initial_questions'] = {}
    if bot_key not in st.session_state['initial_questions']:
        st.session_state['initial_questions'][bot_key] = generate_initial_questions(bot_info)

def handle_enterprise_consultation_bot(bot_info, bot_key):
    vectordb = create_expert_vectordb()
    
    user_input = st.text_area("기업 애로사항을 입력해주세요:")
    
    if st.button("전문가 찾기 및 상담"):
        if user_input:
            relevant_experts = get_relevant_experts(vectordb, user_input)
            
            if not relevant_experts:
                st.write("관련 전문가를 찾을 수 없습니다.")
                return
            
            st.write("관련 전문가 정보:")
            display_expert_table(relevant_experts)
            
            add_message(bot_key, "user", user_input)
            
            expert = relevant_experts[0]  # 가장 관련성 높은 전문가 선택
            
            chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            system_message = SystemMessage(content=f"당신은 {expert['name']}이며, {expert['specialty']} 분야의 전문가입니다. 사용자의 질문에 {expert['specialty']} 전문가로서 답변해주세요.")
            human_message = HumanMessage(content=f"기업 애로사항: {user_input}\n\n위 애로사항에 대해 {expert['specialty']} 전문가로서 조언해주세요.")
            
            messages = [system_message, human_message]
            response = chat(messages)
            bot_response = f"{expert['name']} 전문가의 답변:\n\n{response.content}"
            
            add_message(bot_key, "assistant", bot_response)
            
            with st.chat_message("assistant"):
                st.write(bot_response)
        else:
            st.warning("애로사항을 입력해주세요.")
    
    display_chat_history(bot_key)


def handle_general_bot(bot_info, bot_key):
    handle_pdf_upload_section()
    handle_additional_questions_toggle()
    display_chat_history(bot_key)
    display_initial_questions(bot_key)
    handle_user_input(bot_info, bot_key)

def handle_pdf_upload_section():
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)
    if uploaded_file:
        pdf_text, vector_store = handle_pdf_upload(uploaded_file)
        if pdf_text and vector_store:
            st.session_state['pdf_text'] = pdf_text
            st.session_state['vector_store'] = vector_store
            st.success("PDF가 성공적으로 업로드되고 분석되었습니다.")

def handle_additional_questions_toggle():
    st.session_state['generate_additional_questions'] = st.checkbox(
        "추가 질문 자동 생성", 
        value=st.session_state.get('generate_additional_questions', False)
    )

def display_chat_history(bot_key):
    if 'messages' in st.session_state and bot_key in st.session_state['messages']:
        for message in st.session_state['messages'][bot_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "additional_questions" in message:
                    st.write("추가 질문 제안:")
                    for i, q in enumerate(message["additional_questions"]):
                        if st.button(q, key=f"additional_{bot_key}_{i}"):
                            st.session_state['user_input'] = q
                            st.experimental_rerun()

def display_initial_questions(bot_key):
    if not st.session_state['messages'][bot_key]:
        st.write("추천 시작 질문:")
        for i, q in enumerate(st.session_state['initial_questions'][bot_key]):
            if st.button(q, key=f"initial_{bot_key}_{i}"):
                st.session_state['user_input'] = q
                st.experimental_rerun()

def handle_user_input(bot_info, bot_key):
    user_input = st.chat_input("질문을 입력하세요:")
    
    if user_input or st.session_state.get('user_input'):
        if st.session_state.get('user_input'):
            user_input = st.session_state['user_input']
            del st.session_state['user_input']
        
        process_user_input(user_input, bot_info, bot_key)

def process_user_input(user_input, bot_info, bot_key):
    add_message(bot_key, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in generate_answer_stream(user_input, bot_info):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    if st.session_state.get('generate_additional_questions', False):
        additional_questions = generate_additional_questions_func(full_response)
        add_message(bot_key, "assistant", full_response, additional_questions)
    else:
        add_message(bot_key, "assistant", full_response)
    
    st.experimental_rerun()

def add_message(bot_key, role, content, additional_questions=None):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = {}
    if bot_key not in st.session_state['messages']:
        st.session_state['messages'][bot_key] = []
    message = {"role": role, "content": content}
    if additional_questions:
        message["additional_questions"] = additional_questions
    st.session_state['messages'][bot_key].append(message)


def reset_chat_history(bot_key, bot_info):
    st.session_state['messages'][bot_key] = []
    st.session_state['initial_questions'][bot_key] = generate_initial_questions(bot_info)
    st.experimental_rerun()

def generate_answer_stream(question, bot_info):
    if 'vector_store' in st.session_state and st.session_state['vector_store']:
        pdf_answer, source_docs = get_answer_from_pdf(question, st.session_state['vector_store'])
        llm = ChatOpenAI(streaming=True, temperature=0)
        response = llm.stream(f"Question: {question}\n\nContext from PDF: {pdf_answer}\n\nAnswer:")
    else:
        llm = ChatOpenAI(streaming=True, temperature=0.7)
        response = llm.stream(question)
    
    for chunk in response:
        yield chunk.content

def generate_initial_questions(bot_info):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        """Given the following bot information, generate 3 initial questions that a user might ask to start the conversation:
        Bot Name: {name}
        Bot Introduction: {introduction}
        Bot Personality: {personality}

        Questions:
        1.
        2.
        3."""
    )
    response = chat(prompt.format_messages(
        name=bot_info['name'],
        introduction=bot_info['introduction'],
        personality=bot_info.get('personality', 'Helpful and friendly')
    ))
    questions = response.content.split("\n")[1:]
    return [q.strip()[3:] for q in questions if q.strip()]

def generate_additional_questions_func(content):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        """Given the following content, generate 3 follow-up questions that a user might ask:
        Content: {content}
        
        Questions:
        1.
        2.
        3."""
    )
    response = chat(prompt.format_messages(content=content))
    questions = response.content.split("\n")[1:]
    return [q.strip()[3:] for q in questions if q.strip()]

def create_expert_vectordb():
    experts = [
        {"name": "김전문", "contact": "010-1234-5678", "email": "kim@expert.com", "specialty": "재무 관리"},
        {"name": "이컨설턴트", "contact": "010-2345-6789", "email": "lee@consultant.com", "specialty": "마케팅 전략"},
        {"name": "박박사", "contact": "010-3456-7890", "email": "park@phd.com", "specialty": "기술 혁신"},
        {"name": "정멘토", "contact": "010-4567-8901", "email": "jung@mentor.com", "specialty": "인사 관리"},
        {"name": "최고문", "contact": "010-5678-9012", "email": "choi@advisor.com", "specialty": "법률 자문"}
    ]
    
    documents = [Document(page_content=f"{expert['name']} 전문가는 {expert['specialty']} 분야의 전문가입니다.", 
                          metadata=expert) for expert in experts]
    
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    
    return vectordb

def get_relevant_experts(vectordb, query, k=1):
    relevant_docs = vectordb.similarity_search(query, k=k)
    return [doc.metadata for doc in relevant_docs]

def display_expert_table(experts):
    df = pd.DataFrame(experts)
    df['email'] = df['email'].apply(lambda x: f'<a href="mailto:{x}">{x}</a>')
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

def setup_page():
    st.set_page_config(page_title="티그리스 AI+ 💬", page_icon="💬")
    if 'custom_bots' not in st.session_state:
        st.session_state['custom_bots'] = []
    if 'template_results' not in st.session_state:
        st.session_state['template_results'] = {}
    if 'bot_index' not in st.session_state:
        st.session_state['bot_index'] = None
    if 'base' not in st.session_state:
        st.session_state['base'] = ["사용자 튜토리얼", "프롬프트 메이커", "tmp1"]
    if 'services' not in st.session_state:
        st.session_state['services'] = ["사업계획서 작성", "교과내용 생성"]
    if 'initial_questions' not in st.session_state:
        st.session_state['initial_questions'] = {}
    if 'messages' not in st.session_state:
        st.session_state['messages'] = {}
    if 'generate_additional_questions' not in st.session_state:
        st.session_state['generate_additional_questions'] = False

def display_main_page():
    st.write("원하시는 서비스를 선택하거나 새로운 서비스를 추가하세요.")
    
    display_search_bar()
    display_base_service_templates()
    display_service_templates()
    display_basic_templates()
    bot_create()
    display_custom_bots()
    display_template_results()
    
    if st.button("새 서비스 추가", key="add_new_service_button"):
        st.session_state['page'] = 'select_format'
        st.experimental_rerun()

def display_search_bar():
    search_query = st.text_input("봇 검색 (이름 또는 해시태그)", key="bot_search_input_main")
    if search_query and st.session_state['bot_index']:
        results = st.session_state['bot_index'].similarity_search(search_query, k=10)
        st.write("검색 결과:")
        displayed_bots = set()
        for i, result in enumerate(results):
            bot_name = result.metadata['name']
            hashtags = result.metadata['hashtags']
            if (search_query.lower() in bot_name.lower() or 
                any(search_query.lower() in tag.lower() for tag in hashtags)) and bot_name not in displayed_bots:
                if st.button(f"{bot_name}", key=f"search_result_{bot_name}_{i}"):
                    bot = next((b for b in st.session_state['custom_bots'] if b['name'] == bot_name), None)
                    if bot:
                        st.session_state['page'] = 'chat_with_bot'
                        st.session_state['current_bot'] = bot
                        st.experimental_rerun()
                displayed_bots.add(bot_name)

def display_base_service_templates():
    st.subheader("공통 봇")
    basic_templates = get_basic_templates(base_path)
    basic_templates.sort()  # 가나다 순으로 정렬

    num_templates = len(basic_templates)
    templates_per_row = 4
    num_rows = math.ceil(num_templates / templates_per_row)

    for row in range(num_rows):
        cols = st.columns(templates_per_row)
        for i in range(templates_per_row):
            index = row * templates_per_row + i
            if index < num_templates:
                template = basic_templates[index]
                with cols[i]:
                    if st.button(template, key=f"basic_template_{index}"):
                        bot_info = load_template(base_path, template)
                        if bot_info:
                            st.session_state['current_bot'] = bot_info
                            st.session_state['page'] = 'chat_with_bot'
                            st.experimental_rerun()


def display_service_templates():
    st.subheader("템플릿 봇")
    services = sorted(st.session_state['services'])  # 가나다 순으로 정렬

    template_cols = st.columns(4)
    for i, service in enumerate(services):
        with template_cols[i % 4]:
            if st.button(service, key=f"provided_template_{i}"):
                handle_template_selection(service)

def display_basic_templates():
    st.subheader("대화형 봇")
    basic_templates = get_basic_templates(basic_path)
    basic_templates.sort()  # 가나다 순으로 정렬

    num_templates = len(basic_templates)
    templates_per_row = 4
    num_rows = math.ceil(num_templates / templates_per_row)

    for row in range(num_rows):
        cols = st.columns(templates_per_row)
        for i in range(templates_per_row):
            index = row * templates_per_row + i
            if index < num_templates:
                template = basic_templates[index]
                with cols[i]:
                    if st.button(template, key=f"conversation_template_{index}"):
                        bot_info = load_template(basic_path, template)
                        if bot_info:
                            st.session_state['current_bot'] = bot_info
                            st.session_state['page'] = 'chat_with_bot'
                            st.experimental_rerun()


def handle_template_selection(template):
    if template in get_basic_templates(base_path):
        st.session_state['page'] = 'chat_with_bot'
        st.session_state['current_bot'] = {'name': template}
    else:
        if template == "교과내용 생성":
            st.session_state['page'] = 'curriculum_generation'
        elif template == "사업계획서 작성":
            st.session_state['page'] = 'business_plan_creation'
        else:
            st.session_state['page'] = 'create_template_bot'
        st.session_state['selected_service'] = template
    st.experimental_rerun()

def display_custom_bots():
    if any(bot for bot in st.session_state['custom_bots'] if bot.get('format') == 'basic'):
        st.subheader("기본 형식 봇")
        basic_bots = [bot for bot in st.session_state['custom_bots'] if bot.get('format') == 'basic']
        cols = st.columns(min(3, len(basic_bots)))
        for i, bot in enumerate(basic_bots):
            with cols[i % 3]:
                if st.button(f"{bot['name']}", key=f"basic_{i}"):
                    st.session_state['page'] = 'chat_with_bot'
                    st.session_state['current_bot'] = bot
                    st.experimental_rerun()

def display_template_results():
    if st.session_state['template_results']:
        st.subheader("템플릿 형식 봇")
        template_cols = st.columns(min(3, len(st.session_state['template_results'])))
        for i, (name, result) in enumerate(st.session_state['template_results'].items()):
            with template_cols[i % 3]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"{name}", key=f"template_result_{i}"):
                        st.session_state['page'] = 'view_template_result'
                        st.session_state['current_result'] = result
                        st.experimental_rerun()
                with col2:
                    if st.button("삭제", key=f"delete_result_{i}"):
                        del st.session_state['template_results'][name]
                        st.experimental_rerun()

def bot_create():
    st.subheader("새 서비스 추가")

def select_format_page():
    st.header("새 서비스 추가")
    
    format_option = st.radio(
        "봇 형식을 선택하세요:",
        ('기본 형식', '템플릿 형식')
    )

    if st.button("다음"):
        if format_option == '기본 형식':
            st.session_state['bot_format'] = 'basic'
            st.session_state['page'] = 'create_bot'
        else:
            st.session_state['bot_format'] = 'template'
            st.session_state['page'] = 'create_template_bot'
        st.experimental_rerun()

def view_template_result():
    if 'current_result' not in st.session_state:
        st.error("선택된 결과가 없습니다.")
        return

    result = st.session_state['current_result']
    st.title(f"{result['name']} 결과")

    if 'content' in result:
        if isinstance(result['content'], list):
            for item in result['content']:
                if isinstance(item, dict):
                    st.subheader(item.get('title', '제목 없음'))
                    st.write(item.get('content', '내용 없음'))
                else:
                    st.write(item)
        elif isinstance(result['content'], dict):
            for key, value in result['content'].items():
                st.subheader(key)
                st.write(value)
        else:
            st.write(result['content'])
    else:
        st.write("내용이 없습니다.")

    if st.button("메인 페이지로 돌아가기"):
        st.session_state['page'] = 'main'
        st.experimental_rerun()


