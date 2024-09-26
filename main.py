import streamlit as st
from dotenv import load_dotenv
from ui_components import setup_page, display_main_page, select_format_page, chat_with_bot, view_template_result
from curriculum_generator import curriculum_generator
from course_generator import course_generator
from bot_creator import create_bot, create_template_bot
from business_plan_creator import create_business_plan

load_dotenv()

# 전역 변수로 pages 딕셔너리 정의
pages = {
    'main': display_main_page,
    'select_format': select_format_page,
    'chat_with_bot': lambda: chat_with_bot(st.session_state.get('current_bot')),
    'create_bot': create_bot,
    'create_template_bot': lambda: create_template_bot(st.session_state.get('selected_service')),
    'view_template_result': view_template_result,
    'curriculum_generation': curriculum_generator,
    'business_plan_creation': create_business_plan
}

def main():
    setup_page()
    st.title("티그리스 AI+")
    
    if 'page' not in st.session_state:
        st.session_state['page'] = 'main'
    
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
        # 'messages'와 'initial_questions'를 완전히 초기화하지 않고 유지
        st.session_state.pop('pdf_text', None)
        st.session_state.pop('vector_store', None)
        st.session_state.pop('step', None)
        st.session_state.pop('toc', None)
        st.session_state.pop('hierarchical_toc', None)
        st.experimental_rerun()

if __name__ == "__main__":
    main()