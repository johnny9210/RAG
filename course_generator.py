import streamlit as st
from utils import generate_content
import yaml

def course_generator():
    st.header("교과 내용 생성기")
    
    with open('course_generation_config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    st.header(config['title'])
    main_input = st.text_input(config['main_input_label'])
    
    additional_inputs = {}
    for key, input_config in config['additional_inputs'].items():
        if input_config['type'] == 'text':
            additional_inputs[key] = st.text_input(input_config['label'])
        elif input_config['type'] == 'number':
            additional_inputs[key] = st.number_input(
                input_config['label'],
                min_value=input_config.get('min'),
                max_value=input_config.get('max'),
                value=input_config.get('default')
            )
    
    if st.button("교과내용 생성"):
        try:
            course_content = generate_course_content(config, main_input, additional_inputs)
            if course_content:
                st.session_state['course_content'] = course_content
                st.success("교과내용이 생성되었습니다!")
                st.markdown(course_content)
            else:
                st.error("교과내용 생성에 실패했습니다. 다시 시도해주세요.")
        except Exception as e:
            st.error(f"교과내용 생성 중 오류가 발생했습니다: {str(e)}")

def generate_course_content(config, main_input, additional_inputs):
    prompt = f"""
    강의 주제: {main_input}
    추가 정보:
    {', '.join([f'{k}: {v}' for k, v in additional_inputs.items()])}
    
    위 정보를 바탕으로 다음 형식에 맞춰 교과 내용을 생성해주세요:
    1. 강의 개요 (3-5문장)
    2. 학습 목표 (5-7개의 구체적인 목표)
    3. 주차별 학습 내용 (16주 과정):
    - 각 주차의 주제
    - 각 주차의 간략한 설명 (1-2문장)
    4. 평가 방법 (중간고사, 기말고사, 과제 등)
    
    응답은 마크다운 형식으로 작성해주세요.
    """
    return generate_content(prompt)