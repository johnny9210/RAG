import streamlit as st
from pdf_analyzer import (
    extract_toc_from_pdf,
    generate_toc_from_description,
    parse_toc,
    generate_section_content,
    format_content_for_pdf,
    generate_pdf,
    create_download_link
)

from utils import generate_content  # 파일 상단에 이 import 문 추가
from pdf_utils import process_pdf, create_elasticsearch_store



def create_business_plan():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    steps = {
        1: step_1,
        2: step_2,
        3: step_3,
        
    }
    
    result = steps[st.session_state.step]()
    
    if result == "main_page":
        st.session_state['page'] = 'main'
        st.experimental_rerun()
    elif isinstance(result, dict):
        st.session_state['template_results'][result['name']] = result
        st.success(f"{result['name']} 결과물이 성공적으로 생성되었습니다!")
        st.session_state['page'] = 'main'
        st.experimental_rerun()

def step_1():
    st.subheader("1. PDF 업로드 및 분석")
    
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.pdf_contents = []
        st.session_state.pdf_vector_stores = []
        for uploaded_file in uploaded_files:
            pdf_text = process_pdf(uploaded_file)
            if pdf_text:
                st.success(f"{uploaded_file.name}이 성공적으로 업로드되고 처리되었습니다.")
                st.session_state.pdf_contents.append(pdf_text)
                st.session_state.pdf_vector_stores.append(create_elasticsearch_store(pdf_text))
            else:
                st.error(f"{uploaded_file.name} 처리 중 오류가 발생했습니다.")
    
    additional_fields = ['사업 목적', '추진 배경', '작성 가이드', '작성 요령', '작성 기준']
    for field in additional_fields:
        st.session_state[field] = st.text_area(field.replace('_', ' ').title(), 
                                               value=st.session_state.get(field, ''),
                                               key=f"{field}_input")
    
    # 평가 기준 PDF 업로드
    evaluation_criteria_file = st.file_uploader("평가 기준 PDF를 업로드하세요", type="pdf")
    if evaluation_criteria_file:
        evaluation_criteria_text = process_pdf(evaluation_criteria_file)
        if evaluation_criteria_text:
            st.session_state['evaluation_criteria'] = evaluation_criteria_text
            st.success("평가 기준 PDF가 성공적으로 업로드되고 처리되었습니다.")
        else:
            st.error("평가 기준 PDF 처리 중 오류가 발생했습니다.")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("메인 페이지로"):
            return "main_page"
    with col3:
        if st.button("다음", key="next_to_step_2"):
            st.session_state.step = 2
            st.experimental_rerun()

    return None

def step_2():
    st.subheader("2. 목차 작성")
    
    if 'pdf_contents' in st.session_state and st.session_state.pdf_contents:
        toc = extract_toc_from_pdf("\n".join(st.session_state.pdf_contents))
    else:
        toc = generate_toc_from_description(st.session_state, generate_content)
    
    if 'toc' not in st.session_state:
        st.session_state.toc = toc
    
    st.write("사업 계획서 대제목 및 소제목")
    
    parsed_toc = parse_toc(st.session_state.toc)
    
    for main_item in parsed_toc:
        with st.expander(f"{main_item['number']} {main_item['title']}", expanded=True):
            main_title = st.text_input(f"대제목", value=main_item['title'], key=f"main_{main_item['number']}")
            
            for sub_item in main_item['sub_items']:
                sub_title = st.text_input(f"  {sub_item['number']}", value=sub_item['title'], key=f"sub_{sub_item['number']}")
            
            if 'new_objectives' not in main_item:
                main_item['new_objectives'] = []
            
            for i, objective in enumerate(main_item['new_objectives']):
                st.text_input(f"추가 목표 {i+1}", value=objective, key=f"new_objective_{main_item['number']}_{i}")
            
            if st.button(f"소제목 추가 ({main_item['number']})", key=f"add_objective_{main_item['number']}"):
                main_item['new_objectives'].append("")
                st.experimental_rerun()
    
    if st.button("목차 저장"):
        # Update the session state with the modified TOC
        updated_toc = []
        for main_item in parsed_toc:
            updated_toc.append(f"{main_item['number']} {main_item['title']}")
            for sub_item in main_item['sub_items']:
                updated_toc.append(f"{sub_item['number']} {sub_item['title']}")
            for i, objective in enumerate(main_item['new_objectives'], start=1):
                if objective:  # Only add non-empty objectives
                    updated_toc.append(f"{main_item['number']}.{len(main_item['sub_items'])+i} {objective}")
        st.session_state.toc = updated_toc
        st.success("목차가 저장되었습니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("이전", key="back_to_step_1"):
            st.session_state.step = 1
            st.experimental_rerun()
    with col2:
        if st.button("메인 페이지로"):
            return "main_page"
    with col3:
        if st.button("다음", key="next_to_step_3"):
            if 'toc' not in st.session_state or not st.session_state.toc:
                st.error("다음 단계로 진행하기 전에 목차를 저장해주세요.")
            else:
                st.session_state.step = 3
                st.experimental_rerun()

    return None



def step_3():
    st.subheader("3. 내용 생성")
    
    if 'parsed_toc' not in st.session_state:
        st.session_state.parsed_toc = parse_toc(st.session_state.toc)
    
    for main_section in st.session_state.parsed_toc:
        with st.expander(main_section['title'], expanded=False):
            st.write(f"## {main_section['title']}")
            for sub_section in main_section['sub_items']:
                st.subheader(sub_section['title'])
                if not sub_section.get('content'):
                    if st.button(f"내용 생성: {sub_section['title']}", key=f"gen_{sub_section['title']}"):
                        with st.spinner("내용을 생성 중입니다..."):
                            context = f"{main_section['title']} - {sub_section['title']}"
                            sub_section['content'] = generate_section_content(sub_section['title'], context, st.session_state)
                
                if sub_section.get('content'):
                    sub_section['content'] = st.text_area(
                        "내용",
                        value=sub_section['content'],
                        height=300,
                        key=f"content_{sub_section['title']}"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"평가: {sub_section['title']}", key=f"evaluate_{sub_section['title']}"):
                            evaluation = evaluate_content(sub_section['content'], sub_section['title'], st.session_state)
                            st.write("평가 결과:")
                            st.write(evaluation)
                            sub_section['evaluation'] = evaluation
                    with col2:
                        if st.button(f"내용 재생성: {sub_section['title']}", key=f"regenerate_{sub_section['title']}"):
                            if 'evaluation' in sub_section:
                                with st.spinner("내용을 재생성 중입니다..."):
                                    new_content = regenerate_content(sub_section['content'], sub_section['evaluation'], st.session_state)
                                    sub_section['content'] = new_content
                                    st.success("내용이 재생성되었습니다.")
                                    st.experimental_rerun()
                            else:
                                st.warning("먼저 평가를 수행해주세요.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("이전"):
            st.session_state.step = 2
            st.experimental_rerun()
    with col2:
        if st.button("PDF 생성"):
            content = format_content_for_pdf(st.session_state.parsed_toc)
            pdf_content = generate_pdf(content)
            if pdf_content:
                st.markdown(create_download_link(pdf_content), unsafe_allow_html=True)
            else:
                st.error("PDF 생성에 실패했습니다.")
    with col3:
        if st.button("메인 페이지로 돌아가기"):
            return "main_page"

    return None

def generate_section_content(title, context, session_state):
    prompt = f"""
    사업계획서 섹션 제목: {title}
    컨텍스트: {context}
    사업 목적: {session_state.get('사업 목적', '')}
    추진 배경: {session_state.get('추진 배경', '')}
    작성 가이드: {session_state.get('작성 가이드', '')}
    작성 요령: {session_state.get('작성 요령', '')}
    작성 기준: {session_state.get('작성 기준', '')}
    
    위 정보를 바탕으로 해당 섹션의 내용을 생성해주세요. 
    내용은 구체적이고 관련성 있어야 하며, 최소 200단어 이상으로 작성해 주세요.
    """
    return generate_content(prompt)

def evaluate_content(content, section_title, session_state):
    evaluation_criteria = session_state.get('evaluation_criteria', '')
    
    # 섹션 제목에 따라 관련 평가 기준만 선택
    relevant_criteria = select_relevant_criteria(evaluation_criteria, section_title)
    
    prompt = f"""
    다음 내용을 주어진 기준에 따라 평가해주세요:

    섹션 제목: {section_title}
    
    내용:
    {content}

    관련 평가 기준:
    {relevant_criteria}

    사업 목적: {session_state.get('사업 목적', '')}
    추진 배경: {session_state.get('추진 배경', '')}
    작성 가이드: {session_state.get('작성 가이드', '')}
    작성 요령: {session_state.get('작성 요령', '')}
    작성 기준: {session_state.get('작성 기준', '')}

    각 관련 기준에 대해 1-10점으로 평가하고, 개선점을 제안해주세요.
    평가 시 섹션의 제목과 목적을 고려하여 관련성 높은 기준에 중점을 두세요.
    """
    return generate_content(prompt)

def select_relevant_criteria(evaluation_criteria, section_title):
    # 여기에 섹션 제목에 따라 관련 평가 기준을 선택하는 로직을 구현합니다.
    # 예를 들어, 키워드 매칭이나 규칙 기반 선택 등을 사용할 수 있습니다.
    # 이 예시에서는 간단한 키워드 매칭을 사용합니다.
    
    relevant_keywords = {
        "사업개요": ["목적", "배경", "필요성"],
        "추진전략": ["전략", "방법", "계획"],
        "성과목표": ["목표", "성과", "지표"],
        "추진체계": ["조직", "인력", "역할"],
        "세부추진계획": ["일정", "단계", "세부사항"],
        "소요예산": ["예산", "비용", "재정"],
        "기대효과": ["효과", "영향", "결과"]
    }
    
    selected_criteria = []
    for key, keywords in relevant_keywords.items():
        if any(keyword.lower() in section_title.lower() for keyword in keywords):
            # 해당 키워드에 관련된 평가 기준을 선택
            selected_criteria.extend([line for line in evaluation_criteria.split('\n') if key.lower() in line.lower()])
    
    return "\n".join(selected_criteria) if selected_criteria else evaluation_criteria


def regenerate_content(content, evaluation, session_state):
    prompt = f"""
    다음은 기존 내용과 그에 대한 평가입니다:

    기존 내용:
    {content}

    평가:
    {evaluation}

    다음 기준을 고려하여 내용을 개선해주세요:
    사업 목적: {session_state.get('사업 목적', '')}
    추진 배경: {session_state.get('추진 배경', '')}
    작성 가이드: {session_state.get('작성 가이드', '')}
    작성 요령: {session_state.get('작성 요령', '')}
    작성 기준: {session_state.get('작성 기준', '')}

    평가에서 지적된 부분을 개선하고, 기준에 더 잘 부합하도록 내용을 재작성해주세요.
    """
    return generate_content(prompt)