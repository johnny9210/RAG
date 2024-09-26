import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def generate_pdf(content):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # 여백 설정
    left_margin = 50
    top_margin = height - 50
    line_height = 14
    
    # 폰트 설정
    font_path = "/Users/ilgyun/Documents/langchain/langchain-kr-main/19-Streamlit/03-test/NanumGothic-Regular.ttf"
    font_name = "NanumGothic"
    
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except:
        print(f"NanumGothic 폰트 로딩 실패. 기본 폰트를 사용합니다.")
        font_name = "Helvetica"
    
    # 텍스트를 여러 줄로 나누기
    lines = content.split('\n')
    
    y = top_margin
    for line in lines:
        if line.startswith('# '):  # 대제목
            c.setFont(font_name, 16)
            y -= 20  # 대제목 위 여백
        elif line.startswith('## '):  # 소제목
            c.setFont(font_name, 14)
            y -= 10  # 소제목 위 여백
        else:  # 일반 텍스트
            c.setFont(font_name, 12)
        
        # 텍스트 그리기
        c.drawString(left_margin, y, line)
        y -= line_height
        
        # 페이지가 가득 차면 새 페이지 시작
        if y < 50:
            c.showPage()
            y = top_margin
    
    c.save()
    buffer.seek(0)
    return buffer

def create_download_link(pdf_content):
    b64 = base64.b64encode(pdf_content.getvalue()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="generated_report.pdf">Download PDF</a>'

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

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def answer_question(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return answer


def extract_toc_from_pdf(pdf_content):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """다음은 PDF 문서의 내용입니다. 이 내용을 바탕으로 구조화된 목차를 생성해주세요.
        목차는 다음 규칙을 따라야 합니다:
        1. 최대 3단계의 깊이까지만 생성합니다 (예: 1.1.1).
        2. 각 항목은 간결하고 명확해야 합니다.
        3. 전체 목차는 10개 항목을 넘지 않아야 합니다.
        4. 목차는 문서의 전체 구조를 잘 반영해야 합니다.
        5. 각 목차 항목은 "번호 제목" 형식으로 작성해주세요. (예: "1. 서론", "1.1 연구 배경")

        PDF 내용:
        {pdf_content}

        위 내용을 바탕으로 구조화된 목차를 생성해주세요."""
    )
    response = chat(prompt.format_messages(pdf_content=pdf_content[:2000]))  # 처음 2000자만 사용
    toc = response.content.split('\n')
    return [item.strip() for item in toc if item.strip()]


def generate_toc_from_description(session_state, generate_content_func):
    prompt = f"""
    다음 정보를 바탕으로 사업계획서의 목차를 생성해주세요:
    사업의 목적: {session_state.get('business_purpose', '')}
    추진 배경: {session_state.get('background', '')}
    작성 가이드: {session_state.get('writing_guide', '')}
    작성 요령: {session_state.get('writing_tips', '')}
    작성 기준: {session_state.get('writing_criteria', '')}
    목차는 다음 형식으로 작성해주세요:
    1. 대제목
    1.1. 소제목
    1.1.1. 세부 제목
    목차는 최소 3개의 대제목과 각 대제목 아래 2-3개의 소제목을 포함해야 합니다.
    """
    response = generate_content_func(prompt)
    return [item.strip() for item in response.split('\n') if item.strip()]

def parse_toc(toc):
    hierarchical_toc = []
    current_main_item = None
    
    for item in toc:
        parts = item.split(' ', 1)
        if len(parts) != 2:
            continue  # 올바르지 않은 형식의 항목은 무시
        
        number, title = parts
        number = number.rstrip('.')  # 끝에 있는 점(.) 제거
        depth = len(number.split('.'))
        
        if depth == 1:
            # 대제목
            current_main_item = {'number': number, 'title': title, 'depth': depth - 1, 'sub_items': []}
            hierarchical_toc.append(current_main_item)
        elif depth > 1 and current_main_item:
            # 소제목
            current_main_item['sub_items'].append({'number': number, 'title': title, 'depth': depth - 1})
    
    return hierarchical_toc


def generate_section_content(title, context, generate_content_func):
    prompt = f"""
    제목: {title}
    컨텍스트: {context}
    위 정보를 바탕으로 해당 섹션의 내용을 생성해주세요. 
    최소 200단어 이상으로 구체적이고 관련성 있는 내용을 작성해 주세요.
    """
    return generate_content_func(prompt)

# def format_content_for_pdf(parsed_toc):
#     content = "# 사업계획서\n\n"
#     for main_section in parsed_toc:
#         content += f"# {main_section['title']}\n\n"
#         for sub_section in main_section['subsections']:
#             content += f"## {sub_section['title']}\n\n"
#             content += f"{sub_section.get('content', '')}\n\n"
#     return content

def format_content_for_pdf(parsed_toc):
    content = "# 사업계획서\n\n"
    for main_section in parsed_toc:
        content += f"# {main_section['number']}. {main_section['title']}\n\n"
        if 'content' in main_section:
            content += f"{main_section['content']}\n\n"
        for sub_section in main_section.get('sub_items', []):
            content += f"## {sub_section['number']} {sub_section['title']}\n\n"
            if 'content' in sub_section:
                content += f"{sub_section['content']}\n\n"
    return content