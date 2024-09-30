import os
import json
import base64
import requests
import pymupdf
import tempfile
from PIL import Image
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_teddynote.models import MultiModal
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

from content_generator import generate_content
from dotenv import load_dotenv
load_dotenv()



api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")

class GraphState(TypedDict):
    filepath: str
    filetype: str
    page_numbers: list[int]
    batch_size: int
    split_filepaths: list[str]
    analyzed_files: list[str]
    page_elements: dict[int, dict[str, list[dict]]]
    page_metadata: dict[int, dict]
    page_summary: dict[int, str]
    images: list[str]
    images_summary: list[str]
    tables: list[str]
    tables_summary: dict[int, str]
    texts: list[str]
    texts_summary: list[str]

class LayoutAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key

    def _upstage_layout_analysis(self, input_file):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"ocr": False}
        files = {"document": open(input_file, "rb")}
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/layout-analysis",
            headers=headers,
            data=data,
            files=files,
        )
        if response.status_code == 200:
            output_file = os.path.splitext(input_file)[0] + ".json"
            with open(output_file, "w") as f:
                json.dump(response.json(), f, ensure_ascii=False)
            return output_file
        else:
            raise ValueError(f"API request failed. Status code: {response.status_code}")

    def execute(self, input_file):
        return self._upstage_layout_analysis(input_file)
    
def split_pdf(state: GraphState):
    filepath = state["filepath"]
    batch_size = state["batch_size"]
    input_pdf = pymupdf.open(filepath)
    num_pages = len(input_pdf)
    ret = []
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1
        input_file_basename = os.path.splitext(filepath)[0]
        output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)
    input_pdf.close()
    return GraphState(split_filepaths=ret)

def analyze_layout(state: GraphState):
    split_files = state["split_filepaths"]
    analyzer = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))
    analyzed_files = []
    for file in split_files:
        analyzed_files.append(analyzer.execute(file))
    return GraphState(analyzed_files=sorted(analyzed_files))

def extract_start_end_page(filename):
    file_name = os.path.basename(filename)
    file_name_parts = file_name.split("_")
    if len(file_name_parts) >= 3:
        start_page = int(file_name_parts[-2])
        end_page = int(file_name_parts[-1].split(".")[0])
    else:
        start_page, end_page = 0, 0
    return start_page, end_page

def extract_page_metadata(state: GraphState):
    json_files = state["analyzed_files"]
    page_metadata = dict()
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        start_page, _ = extract_start_end_page(json_file)
        for element in data["metadata"]["pages"]:
            original_page = int(element["page"])
            relative_page = start_page + original_page - 1
            metadata = {
                "size": [
                    int(element["width"]),
                    int(element["height"]),
                ],
            }
            page_metadata[relative_page] = metadata
    return GraphState(page_metadata=page_metadata)

def extract_page_elements(state: GraphState):
    json_files = state["analyzed_files"]
    page_elements = dict()
    element_id = 0
    for json_file in json_files:
        start_page, _ = extract_start_end_page(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)
        for element in data["elements"]:
            original_page = int(element["page"])
            relative_page = start_page + original_page - 1
            if relative_page not in page_elements:
                page_elements[relative_page] = []
            element["id"] = element_id
            element_id += 1
            element["page"] = relative_page
            page_elements[relative_page].append(element)
    return GraphState(page_elements=page_elements)

def extract_tag_elements_per_page(state: GraphState):
    page_elements = state["page_elements"]
    parsed_page_elements = dict()
    for key, page_element in page_elements.items():
        image_elements = []
        table_elements = []
        text_elements = []
        for element in page_element:
            if element["category"] == "figure":
                image_elements.append(element)
            elif element["category"] == "table":
                table_elements.append(element)
            else:
                text_elements.append(element)
        parsed_page_elements[key] = {
            "image_elements": image_elements,
            "table_elements": table_elements,
            "text_elements": text_elements,
            "elements": page_element,
        }
    return GraphState(page_elements=parsed_page_elements)

def generate_pdf(content):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left_margin = 50
    top_margin = height - 50
    line_height = 14
    font_name = "Helvetica"
    lines = content.split('\n')
    y = top_margin
    for line in lines:
        if line.startswith('# '):
            c.setFont(font_name, 16)
            y -= 20
        elif line.startswith('## '):
            c.setFont(font_name, 14)
            y -= 10
        else:
            c.setFont(font_name, 12)
        c.drawString(left_margin, y, line)
        y -= line_height
        if y < 50:
            c.showPage()
            y = top_margin
    c.save()
    buffer.seek(0)
    return buffer

def create_download_link(pdf_content):
    b64 = base64.b64encode(pdf_content.getvalue()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="generated_report.pdf">Download PDF</a>'

prompt = PromptTemplate.from_template(
    """Please summarize the sentence according to the following REQUEST.
    
REQUEST:
1. Summarize the main points in bullet points.
2. Write the summary in same language as the context.
3. DO NOT translate any technical terms.
4. DO NOT include any unnecessary information.
5. Summary must include important entities, numerical values.

CONTEXT:
{context}

SUMMARY:
"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
text_summary_chain = create_stuff_documents_chain(llm, prompt)

def create_text_summary(state: GraphState):
    texts = state["texts"]
    text_summary = dict()
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])
    inputs = [
        {"context": [Document(page_content=text)]} for page_num, text in sorted_texts
    ]
    summaries = text_summary_chain.batch(inputs)
    for page_num, summary in enumerate(summaries):
        text_summary[page_num] = summary
    return GraphState(text_summary=text_summary)

@chain
def extract_image_summary(data_batches):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=api_key)
    system_prompt = """You are an expert in extracting useful information from IMAGE.
    With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."""
    image_paths = []
    system_prompts = []
    user_prompts = []
    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["image"]
        user_prompt_template = f"""Here is the context related to the image: {context}

###

Output Format:

<image>
<title>
<summary>
<entities> 
</image>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)
    multimodal_llm = MultiModal(llm)
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer

def create_image_summary(state: GraphState):
    if not state.get("image_summary_data_batches"):
        return GraphState(image_summary={})

    image_summaries = extract_image_summary.invoke(
        state["image_summary_data_batches"],
    )
    image_summary_output = dict()
    for data_batch, image_summary in zip(
        state["image_summary_data_batches"], image_summaries
    ):
        image_summary_output[data_batch["id"]] = image_summary
    return GraphState(image_summary=image_summary_output)

class ImageCropper:
    @staticmethod
    def pdf_to_image(pdf_file, page_num, dpi=300):
        doc = pymupdf.open(pdf_file)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    @staticmethod
    def normalize_coordinates(coordinates, output_page_size):
        x_values = [coord["x"] for coord in coordinates]
        y_values = [coord["y"] for coord in coordinates]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)
        return (
            x1 / output_page_size[0],
            y1 / output_page_size[1],
            x2 / output_page_size[0],
            y2 / output_page_size[1],
        )

    @staticmethod
    def crop_image(img, coordinates, output_file):
        img_width, img_height = img.size
        x1, y1, x2, y2 = [
            int(coord * dim)
            for coord, dim in zip(coordinates, [img_width, img_height] * 2)
        ]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(output_file)
        print(f"Cropped image saved to: {output_file}")

def crop_image(state: GraphState, save_dir: str):
    pdf_file = state["filepath"]
    page_numbers = state["page_numbers"]
    cropped_images = dict()

    for page_num in page_numbers:
        pdf_image = ImageCropper.pdf_to_image(pdf_file, page_num)
        for element in state["page_elements"][page_num]["image_elements"]:
            if element["category"] == "figure":
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["bounding_box"], state["page_metadata"][page_num]["size"]
                )
                output_file = os.path.join(save_dir, f"image_{element['id']}.png")
                ImageCropper.crop_image(pdf_image, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"Saved image to: {output_file}")

    return GraphState(images=cropped_images)

def crop_table(state: GraphState, save_dir: str):
    pdf_file = state["filepath"]
    page_numbers = state["page_numbers"]
    cropped_tables = dict()

    for page_num in page_numbers:
        pdf_image = ImageCropper.pdf_to_image(pdf_file, page_num)
        for element in state["page_elements"][page_num]["table_elements"]:
            if element["category"] == "table":
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["bounding_box"], state["page_metadata"][page_num]["size"]
                )
                output_file = os.path.join(save_dir, f"table_{element['id']}.png")
                ImageCropper.crop_image(pdf_image, normalized_coordinates, output_file)
                cropped_tables[element["id"]] = output_file
                print(f"Saved table to: {output_file}")

    return GraphState(tables=cropped_tables)

def extract_page_text(state: GraphState):
    page_numbers = state["page_numbers"]
    extracted_texts = dict()
    for page_num in page_numbers:
        extracted_texts[page_num] = ""
        for element in state["page_elements"][page_num]["text_elements"]:
            extracted_texts[page_num] += element["text"]
    return GraphState(texts=extracted_texts)

def create_image_summary_data_batches(state: GraphState):
    data_batches = []
    page_numbers = sorted(list(state["page_elements"].keys()))
    for page_num in page_numbers:
        text = state.get("text_summary", {}).get(page_num, "")
        for image_element in state["page_elements"][page_num].get("image_elements", []):
            image_id = int(image_element["id"])
            if image_id in state.get("images", {}):
                data_batches.append({
                    "image": state["images"][image_id],
                    "text": text,
                    "page": page_num,
                    "id": image_id,
                })
    return GraphState(image_summary_data_batches=data_batches)

def create_table_summary_data_batches(state: GraphState):
    data_batches = []
    page_numbers = sorted(list(state["page_elements"].keys()))
    for page_num in page_numbers:
        text = state["text_summary"][page_num]
        for table_element in state["page_elements"][page_num]["table_elements"]:
            table_id = int(table_element["id"])
            data_batches.append(
                {
                    "table": state["tables"][table_id],
                    "text": text,
                    "page": page_num,
                    "id": table_id,
                }
            )
    return GraphState(table_summary_data_batches=data_batches)

@chain
def extract_table_summary(data_batches):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=api_key)
    system_prompt = "You are an expert in extracting useful information from TABLE. With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."
    image_paths = []
    system_prompts = []
    user_prompts = []
    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["table"]
        user_prompt_template = f"""Here is the context related to the image of table: {context}

###

Output Format:

<table>
<title>
<table_summary>
<key_entities> 
<data_insights>
</table>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)
    multimodal_llm = MultiModal(llm)
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer

def create_table_summary(state: GraphState):
    table_summaries = extract_table_summary.invoke(
        state["table_summary_data_batches"],
    )
    table_summary_output = dict()
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_summaries
    ):
        table_summary_output[data_batch["id"]] = table_summary
    return GraphState(table_summary=table_summary_output)

@chain
def table_markdown_extractor(data_batches):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=api_key)
    system_prompt = "You are an expert in converting image of the TABLE into markdown format. Be sure to include all the information in the table. DO NOT narrate, just answer in markdown format."
    image_paths = []
    system_prompts = []
    user_prompts = []
    for data_batch in data_batches:
        image_path = data_batch["table"]
        user_prompt_template = """DO NOT wrap your answer in ```markdown``` or any XML tags.

###

Output Format:

<table_markdown>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)
    multimodal_llm = MultiModal(llm)
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer

def create_table_markdown(state: GraphState):
    table_markdowns = table_markdown_extractor.invoke(
        state["table_summary_data_batches"],
    )
    table_markdown_output = dict()
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_markdowns
    ):
        table_markdown_output[data_batch["id"]] = table_summary
    return GraphState(table_markdown=table_markdown_output)

# def analyze_pdf(uploaded_file, batch_size: int = 10):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.getvalue())
#         temp_file_path = temp_file.name

#     state = GraphState(filepath=temp_file_path, batch_size=batch_size)
#     state.update(split_pdf(state))
#     state.update(analyze_layout(state))
#     state.update(extract_page_metadata(state))
#     state.update(extract_page_elements(state))
#     state.update(extract_tag_elements_per_page(state))
    
#     state['page_numbers'] = list(state['page_elements'].keys())
    
#     state.update(extract_page_text(state))
#     state.update(create_text_summary(state))
    
#     # 이미지 처리 부분 추가
#     state.update(crop_image(state))
#     state.update(create_image_summary_data_batches(state))
    
#     # 이미지 요약 생성 (이미지가 있는 경우에만)
#     if 'image_summary_data_batches' in state and state['image_summary_data_batches']:
#         state.update(create_image_summary(state))
#     else:
#         print("No images found in the PDF.")

#     os.unlink(temp_file_path)

#     return state  


import os
import shutil
import tempfile

def analyze_pdf(uploaded_file, batch_size: int = 10):
    # 현재 스크립트의 절대 경로를 기준으로 저장 디렉토리 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "image_tmp")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 임시 파일을 원하는 위치로 복사
        dest_file_path = os.path.join(save_dir, uploaded_file.name)
        shutil.copy2(tmp_file_path, dest_file_path)
        print(f"File copied to: {dest_file_path}")

        # 파일이 성공적으로 복사되었는지 확인
        if os.path.exists(dest_file_path):
            print(f"File exists at: {dest_file_path}")
            print(f"File size: {os.path.getsize(dest_file_path)} bytes")
        else:
            print(f"File does not exist at: {dest_file_path}")
            return None

        # 여기서부터 PDF 분석 로직 시작
        state = GraphState(filepath=dest_file_path, batch_size=batch_size)
        state.update(split_pdf(state))
        state.update(analyze_layout(state))
        state.update(extract_page_metadata(state))
        state.update(extract_page_elements(state))
        state.update(extract_tag_elements_per_page(state))

        state['page_numbers'] = list(state['page_elements'].keys())

        state.update(extract_page_text(state))
        state.update(create_text_summary(state))

        # 이미지 처리 부분
        state.update(crop_image(state, save_dir))
        state.update(create_image_summary_data_batches(state))

        # 표 처리 부분
        state.update(crop_table(state, save_dir))
        state.update(create_table_summary_data_batches(state))

        return state

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
    finally:
        # 임시 파일 삭제
        os.unlink(tmp_file_path)

def generate_toc_from_description(session_state, generate_content):
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
    response = generate_content(prompt)
    return [item.strip() for item in response.split('\n') if item.strip()]

def extract_toc_from_pdf(pdf_content):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
    prompt = ChatPromptTemplate.from_template(
        """다음은 PDF 문서의 내용입니다. 이 내용을 바탕으로 구조화된 목차를 생성해주세요.
        목차는 다음 규칙을 따라야 합니다:
        1. 최대 2단계의 깊이까지만 생성합니다 (예: 1.1).
        2. 각 항목은 간결하고 명확해야 합니다.
        3. 전체 목차는 10개 항목을 넘지 않아야 합니다.
        4. 목차는 문서의 전체 구조를 잘 반영해야 합니다.
        5. 각 목차 항목은 "번호 제목" 형식으로 작성해주세요. (예: "1. 서론", "1.1 연구 배경")

        PDF 내용:
        {pdf_content}

        위 내용을 바탕으로 구조화된 목차를 생성해주세요."""
    )
    response = chat(prompt.format_messages(pdf_content=pdf_content[:2000]))
    toc = response.content.split('\n')
    return [item.strip() for item in toc if item.strip()]

def parse_toc(toc):
    hierarchical_toc = []
    current_main_item = None
    for item in toc:
        parts = item.split(' ', 1)
        if len(parts) != 2:
            continue
        number, title = parts
        number = number.rstrip('.')
        depth = len(number.split('.'))
        if depth == 1:
            current_main_item = {'number': number, 'title': title, 'depth': depth - 1, 'sub_items': []}
            hierarchical_toc.append(current_main_item)
        elif depth > 1 and current_main_item:
            current_main_item['sub_items'].append({'number': number, 'title': title, 'depth': depth - 1})
    return hierarchical_toc

def generate_toc_from_description(session_state, generate_content):
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
    목차는 최소 3개의 대제목과 각 대제목 아래 2-3개의 소제목을 포함해야 합니다.
    """
    response = generate_content(prompt)
    return [item.strip() for item in response.split('\n') if item.strip()]

def generate_section_content(title, context, session_state):
    prompt = f"""
    제목: {title}
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