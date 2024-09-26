# curriculum_generator.py

import streamlit as st
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

async def generate_content_async(prompt):
    chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    message = HumanMessage(content=prompt)
    response = await chat.agenerate([[message]])
    return response.generations[0][0].text

async def generate_week_content_async(week, department, course_name, target_year, semester, course_objective, midterm_week, final_week):
    week_str = f"{week}주차"
    if week == midterm_week:
        return "제목: 중간고사\n목표:\n- 중간고사 실시\n내용: 중간고사 시험"
    elif week == final_week:
        return "제목: 기말고사\n목표:\n- 기말고사 실시\n내용: 기말고사 시험"
    
    prompt = f"""
    {department} 학과의 {course_name} 강의 {week_str} 내용을 생성해주세요.
    대상: {target_year} {semester}
    전체 강의 목표: {course_objective}
    
    이 주차는 전체 16주 과정 중 {week_str}에 해당합니다. 전체 강의 목표와 진행 상황을 고려하여 적절한 내용을 생성해주세요.

    다음 형식으로 응답해주세요:
    제목: [주차 제목]
    목표:
    - [목표1]
    - [목표2]
    내용: [간략한 설명]
    """
    return await generate_content_async(prompt)


async def generate_curriculum_async(department, course_name, target_year, semester, course_objective, midterm_week, final_week):
    tasks = []
    for week in range(1, 17):
        tasks.append(generate_week_content_async(
            week, department, course_name, target_year, semester, course_objective, midterm_week, final_week
        ))
    
    results = await asyncio.gather(*tasks)
    return results

def parse_week_content(week_content):
    lines = week_content.split('\n')
    title = "제목 없음"
    objectives = []
    content = "내용 없음"

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("제목:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("목표:"):
            current_section = "objectives"
        elif line.startswith("내용:"):
            current_section = "content"
            content = line.split(":", 1)[1].strip()
        elif current_section == "objectives" and line.startswith("-"):
            objectives.append(line.strip("- "))
        elif current_section == "content" and content == "내용 없음":
            content = line

    return {
        "title": title,
        "objectives": objectives or ["목표 없음"],
        "content": content
    }

async def generate_lesson_plan_async(week, objective, course_name, target_year, semester):
    prompt = f"""
    {course_name} 강의 {week}주차의 다음 학습 목표에 대한 상세 교안을 작성해주세요:
    
    학습 목표: {objective}
    대상: {target_year} {semester}
    
    교안에 포함될 내용:
    1. 도입 (5-10분)
       - 학생들의 관심을 끌 수 있는 구체적인 질문이나 활동 제안
       - 이전 수업 내용과의 연계성 설명
    2. 주요 내용 설명 (30-40분)
       - 핵심 개념에 대한 상세한 설명과 정의
       - 실제 사례나 응용 예시 (최소 2개)
       - 학생 이해도 확인을 위한 중간 질문들 (최소 3개)
    3. 학생 활동 또는 토론 (10-15분)
       - 구체적인 그룹 활동이나 개별 활동 지침
       - 토론 주제 및 진행 방법 상세 설명
    4. 정리 및 요약 (5-10분)
       - 주요 내용에 대한 간결한 요약 (3-5개의 핵심 포인트)
       - 다음 수업 주제 소개 및 연계성 설명

    각 섹션에 대해 시간 배분, 교수 지침, 및 필요한 준비물을 명시해주세요.
    """
    return await generate_content_async(prompt)

def curriculum_generator():
    st.subheader("교과내용 생성기")
    
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    if st.session_state.step == 1:
        st.session_state.department = st.text_input("강의하고자 하는 학과를 입력하세요:")
        st.session_state.course_name = st.text_input("강의명을 입력하세요:")
        st.session_state.target_year = st.selectbox("대상 학년을 선택하세요:", ["1학년", "2학년", "3학년", "4학년"])
        st.session_state.semester = st.selectbox("학기를 선택하세요:", ["1학기", "2학기"])
        st.session_state.midterm_week = st.number_input("중간고사 실시 주차:", min_value=1, max_value=16, value=8)
        st.session_state.final_week = st.number_input("기말고사 실시 주차:", min_value=1, max_value=16, value=16)
        st.session_state.course_objective = st.text_area("전체 강의 목표를 입력하세요:")

        if st.button("다음"):
            with st.spinner("교과내용을 생성 중입니다..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(generate_curriculum_async(
                    st.session_state.department,
                    st.session_state.course_name,
                    st.session_state.target_year,
                    st.session_state.semester,
                    st.session_state.course_objective,
                    st.session_state.midterm_week,
                    st.session_state.final_week
                ))
                st.session_state.curriculum = {}
                for week, content in enumerate(results, 1):
                    week_str = f"{week}주차"
                    st.session_state.curriculum[week_str] = parse_week_content(content)
            st.success("교과내용이 생성되었습니다.")
            st.session_state.step = 2
            st.experimental_rerun()

    elif st.session_state.step == 2:
        st.write(f"과목명: {st.session_state.course_name}")
        st.write("주차별 수업 내용 및 목표")

        for week in range(1, 17):
            week_str = f"{week}주차"
            week_data = st.session_state.curriculum.get(week_str, {
                "title": "",
                "objectives": [""],
                "content": ""
            })

            with st.expander(f"{week_str}: {week_data['title']}", expanded=False):
                if week == st.session_state.midterm_week or week == st.session_state.final_week:
                    st.write(f"{week_data['title']} 주간입니다.")
                else:
                    week_data['title'] = st.text_input(f"{week_str} 제목", value=week_data['title'], key=f"title_{week}")
                    
                    st.write("학습 목표")
                    for i, objective in enumerate(week_data['objectives']):
                        week_data['objectives'][i] = st.text_input(f"학습 목표 {i+1}", value=objective, key=f"objective_{week}_{i}")
                    
                    if st.button(f"학습 목표 추가 (주차 {week})", key=f"add_objective_{week}"):
                        week_data['objectives'].append("")
                        st.experimental_rerun()


            st.session_state.curriculum[week_str] = week_data

        if st.button("다음"):
            st.session_state.step = 3
            st.experimental_rerun()

    elif st.session_state.step == 3:
        st.write("교과내용 상세 정보")
        
        for week in range(1, 17):
            week_str = f"{week}주차"
            week_data = st.session_state.curriculum.get(week_str, {
                "title": "",
                "objectives": [],
                "content": "",
                "lesson_plans": {}
            })
            
            with st.expander(f"{week_str}: {week_data['title']}", expanded=False):
                if week == st.session_state.midterm_week or week == st.session_state.final_week:
                    st.write(f"{week_data['title']} 주간입니다.")
                else:
                    week_data['title'] = st.text_input(f"{week_str} 제목", value=week_data['title'], key=f"title_{week}")
                    
                    st.write("학습 목표 및 교안")
                    objectives = week_data.get('objectives', [])
                    for i, objective in enumerate(objectives):
                        st.write(f"학습 목표 {i+1}")
                        objective = st.text_area(f"목표 내용", value=objective, key=f"objective_{week}_{i}")
                        objectives[i] = objective
                        
                        lesson_plan_key = f"{week_str}_objective_{i}"
                        lesson_plans = week_data.get('lesson_plans', {})
                        lesson_plan = lesson_plans.get(lesson_plan_key, "")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            lesson_plan = st.text_area(f"교안 {i+1}", value=lesson_plan, key=f"lesson_plan_{week}_{i}", height=300)
                        with col2:
                            if st.button(f"교안 생성", key=f"generate_lesson_plan_{week}_{i}"):
                                with st.spinner("교안을 생성 중입니다..."):
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    new_lesson_plan = loop.run_until_complete(generate_lesson_plan_async(
                                        week_str,
                                        objective,
                                        st.session_state.course_name,
                                        st.session_state.target_year,
                                        st.session_state.semester
                                    ))
                                    lesson_plans[lesson_plan_key] = new_lesson_plan
                                    week_data['lesson_plans'] = lesson_plans
                                    st.session_state.curriculum[week_str] = week_data
                                st.success("교안이 생성되었습니다.")
                                st.experimental_rerun()
                        
                        lesson_plans[lesson_plan_key] = lesson_plan
                    
                    week_data['objectives'] = objectives
                    week_data['lesson_plans'] = lesson_plans
                
                st.session_state.curriculum[week_str] = week_data