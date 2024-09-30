# main.py
import streamlit as st
from dotenv import load_dotenv
from ui_components import display_main_page, select_format_page, chat_with_bot, view_template_result, display_prompt_generator
from curriculum_generator import curriculum_generator
from bot_creator import create_bot, create_template_bot
from business_plan_creator import create_business_plan
from pdf_analyzer import analyze_pdf
from mongodb_utils import insert_json_files_to_mongodb
import os

# Streamlit 페이지 설정을 가장 먼저 실행
st.set_page_config(page_title="티그리스 AI+ 💬", page_icon="💬")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))  # 디버깅을 위해 추가

# JSON 파일을 MongoDB에 삽입
json_directory = "/Users/ilgyun/Documents/langchain/langchain-kr-main/19-Streamlit/03-test/base_prompts"
insert_json_files_to_mongodb(json_directory)

# 세션 상태 초기화
if 'services' not in st.session_state:
    st.session_state['services'] = ["사업계획서 작성", "교과내용 생성"]
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'
if 'custom_bots' not in st.session_state:
    st.session_state['custom_bots'] = []
if 'template_results' not in st.session_state:
    st.session_state['template_results'] = {}
if 'bot_index' not in st.session_state:
    st.session_state['bot_index'] = None
if 'base' not in st.session_state:
    st.session_state['base'] = ["사용자 튜토리얼", "프롬프트 메이커", "tmp1"]
if 'initial_questions' not in st.session_state:
    st.session_state['initial_questions'] = {}
if 'messages' not in st.session_state:
    st.session_state['messages'] = {}
if 'generate_additional_questions' not in st.session_state:
    st.session_state['generate_additional_questions'] = False

# 전역 변수로 pages 딕셔너리 정의
pages = {
    'main': display_main_page,
    'select_format': select_format_page,
    'chat_with_bot': lambda: chat_with_bot(st.session_state.get('current_bot')),
    'create_bot': create_bot,
    'create_template_bot': lambda: create_template_bot(st.session_state.get('selected_service')),
    'view_template_result': view_template_result,
    'curriculum_generation': curriculum_generator,
    'business_plan_creation': create_business_plan,
    'pdf_analysis': analyze_pdf,  # 새로 추가된 페이지
    'prompt_generator': display_prompt_generator,

}

def main():
    current_page = st.session_state['page']
    
    if current_page == 'chat_with_bot':
        chat_container = st.container()
        input_container = st.container()
        
        with chat_container:
            pages[current_page]()
        
        with input_container:
            st.markdown("---")
    else:
        pages[current_page]()
    
    if st.button("메인 페이지로 돌아가기", key="return_to_main"):
        st.session_state['page'] = 'main'
        st.session_state.pop('current_bot', None)
        st.session_state.pop('pdf_text', None)
        st.session_state.pop('vector_store', None)
        st.session_state.pop('step', None)
        st.session_state.pop('toc', None)
        st.session_state.pop('hierarchical_toc', None)
        st.experimental_rerun()

if __name__ == "__main__":
    main()