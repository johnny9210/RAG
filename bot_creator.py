# bot_creator.py
import streamlit as st
from pdf_analyzer import (
    process_pdf, 
    extract_toc_from_pdf,
    parse_toc   
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def create_bot():
    if st.session_state.get('bot_format') == 'basic':
        create_basic_bot()
    elif st.session_state.get('bot_format') == 'template':
        create_template_bot(st.session_state.get('selected_service'))
    else:
        st.error("Unsupported bot format")


def create_basic_bot():
    st.header("기본 봇 생성")
    
    if 'basic_bot_info' not in st.session_state:
        st.session_state.basic_bot_info = {
            'name': '',
            'introduction': '',
            'tasks': '',
            'personality': '',
            'rules': '',
            'hashtags': []
        }
    
    st.session_state.basic_bot_info['name'] = st.text_input("봇 이름", value=st.session_state.basic_bot_info['name'])
    st.session_state.basic_bot_info['introduction'] = st.text_area("소개", value=st.session_state.basic_bot_info['introduction'])
    st.session_state.basic_bot_info['tasks'] = st.text_area("수행할 작업", value=st.session_state.basic_bot_info['tasks'])
    st.session_state.basic_bot_info['personality'] = st.text_area("성격", value=st.session_state.basic_bot_info['personality'])
    st.session_state.basic_bot_info['rules'] = st.text_area("규칙", value=st.session_state.basic_bot_info['rules'])
    
    new_hashtag = st.text_input("새 해시태그 (엔터로 추가)")
    if new_hashtag:
        st.session_state.basic_bot_info['hashtags'].append(new_hashtag)
        st.experimental_rerun()
    
    st.write("현재 해시태그:", ', '.join(st.session_state.basic_bot_info['hashtags']))
    
    if st.button("봇 생성"):
        new_bot = st.session_state.basic_bot_info.copy()
        new_bot['format'] = 'basic'
        st.session_state['custom_bots'].append(new_bot)
        st.success(f"{new_bot['name']} 봇이 생성되었습니다!")
        st.session_state['page'] = 'main'
        st.experimental_rerun()


def create_template_bot(service):
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    steps = {
        1: step_1,
        2: step_2,
        3: step_3
    }
    
    result = steps[st.session_state.step](service)
    
    if st.button("메인 페이지로 돌아가기"):
        st.session_state['page'] = 'main'
        st.session_state.pop('toc', None)
        st.session_state.pop('parsed_toc', None)
        st.session_state.pop('step', None)
        st.experimental_rerun()


def step_1(service):
    st.write("## 1. PDF 업로드 및 기본 정보 입력")
    
    # 기본 정보 입력
    st.session_state.business_info = {
        'purpose': st.text_area("사업의 목적"),
        'background': st.text_area("추진 배경"),
        'writing_guide': st.text_area("작성 가이드"),
        'writing_tips': st.text_area("작성 요령"),
        'writing_criteria': st.text_area("작성 기준")
    }
    
    uploaded_file = st.file_uploader("사업계획서 PDF 파일을 업로드하세요", type="pdf")
    
    if uploaded_file:
        pdf_text = process_pdf(uploaded_file)
        if pdf_text:
            toc = extract_toc_from_pdf(pdf_text)
            if toc:
                st.session_state.toc = toc
                st.session_state.pdf_text = pdf_text
                st.success("PDF가 성공적으로 업로드되고 목차가 추출되었습니다.")
                if st.button("다음 단계로"):
                    st.session_state.step = 2
                    st.experimental_rerun()
            else:
                st.error("목차 추출에 실패했습니다. 다시 시도해 주세요.")
        else:
            st.error("PDF 처리 중 오류가 발생했습니다.")

def step_2(service):
    st.write("## 2. 목차 확인 및 수정")
    
    if 'toc' not in st.session_state:
        st.error("추출된 목차가 없습니다. 이전 단계로 돌아가 PDF를 업로드해주세요.")
        return
    
    toc = st.session_state.toc
    
    st.write("현재 목차:")
    for i, item in enumerate(toc):
        toc[i] = st.text_input(f"항목 {i+1}", value=item)
    
    if st.button("목차 확정 및 다음 단계로"):
        st.session_state.toc = toc
        st.session_state.step = 3
        st.experimental_rerun()

def step_3(service):
    st.write("## 3. 내용 생성")
    
    if 'toc' not in st.session_state:
        st.error("목차 정보가 없습니다. 이전 단계로 돌아가 확인해주세요.")
        return
    
    if 'parsed_toc' not in st.session_state:
        st.session_state.parsed_toc = parse_toc(st.session_state.toc)
    
    for item in st.session_state.parsed_toc:
        if item['depth'] == 0:
            st.markdown(f"### {item['number']}. {item['title']}")
        else:
            with st.expander(f"{item['number']} {item['title']}", expanded=True):
                st.markdown(f"**{item['number']} {item['title']}**")
                
                content_placeholder = st.empty()
                content_placeholder.text_area(
                    "본문 내용",
                    value=item['content'],
                    height=200,
                    key=f"content_{item['number']}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("내용 생성", key=f"generate_{item['number']}"):
                        with st.spinner("내용 생성 중..."):
                            item['content'] = generate_content_with_criteria(
                                f"{item['number']} {item['title']}",
                                st.session_state.pdf_text,
                                st.session_state.business_info
                            )
                            content_placeholder.text_area(
                                "본문 내용",
                                value=item['content'],
                                height=200,
                                key=f"new_content_{item['number']}"
                            )
                            st.success("내용이 생성되었습니다.")
                with col2:
                    if st.button("내용 저장", key=f"save_{item['number']}"):
                        item['content'] = content_placeholder.text_area(
                            "본문 내용",
                            value=item['content'],
                            height=200,
                            key=f"save_content_{item['number']}"
                        )
                        st.success("내용이 저장되었습니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("이전"):
            st.session_state.step = 2
            st.experimental_rerun()
    with col2:
        if st.button("PDF 생성"):
            # PDF 생성 로직 구현 필요
            st.info("PDF 생성 기능은 아직 구현되지 않았습니다.")
    with col3:
        if st.button("완료"):
            st.success("사업계획서 생성이 완료되었습니다.")


def generate_content_with_criteria(title, context, criteria):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        """다음 정보를 바탕으로 내용을 생성해주세요:
        제목: {title}
        컨텍스트: {context}
        사업 목적: {business_purpose}
        추진 배경: {background}
        작성 가이드: {writing_guide}
        작성 요령: {writing_tips}
        작성 기준: {writing_criteria}

        위 정보를 모두 고려하여 {title}에 대한 내용을 생성해주세요. 
        최소 200단어 이상으로 구체적이고 관련성 있는 내용을 작성해 주세요."""
    )
    
    response = chat(prompt.format_messages(
        title=title,
        context=context[:1000],  # 컨텍스트의 일부만 사용
        business_purpose=criteria['purpose'],
        background=criteria['background'],
        writing_guide=criteria['writing_guide'],
        writing_tips=criteria['writing_tips'],
        writing_criteria=criteria['writing_criteria']
    ))
    
    return response.content


def evaluate_content(content, criteria):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """다음 내용을 주어진 기준에 따라 평가해주세요:

        내용:
        {content}

        평가 기준:
        사업 목적: {business_purpose}
        추진 배경: {background}
        작성 가이드: {writing_guide}
        작성 요령: {writing_tips}
        작성 기준: {writing_criteria}

        각 기준에 대해 1-10점으로 평가하고, 개선점을 제안해주세요."""
    )
    
    response = chat(prompt.format_messages(
        content=content,
        business_purpose=criteria['사업 목적'],
        background=criteria['추진 배경'],
        writing_guide=criteria['작성 가이드'],
        writing_tips=criteria['작성 요령'],
        writing_criteria=criteria['작성 기준']
    ))
    
    return response.content


