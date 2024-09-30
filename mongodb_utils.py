# mongodb_utils.py

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import json
import os
import streamlit as st

def connect_to_mongodb():
    try:
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        db = client['bot_templates']
        # st.success("MongoDB에 성공적으로 연결되었습니다.")
        return db
    except ConnectionFailure as e:
        st.error(f"MongoDB 연결 실패: {str(e)}")
        return None

def get_all_template_names(db):
    if db is not None:
        collection = db['templates']
        templates = list(collection.find({}, {'name': 1, '_id': 0}))
        # st.write(f"데이터베이스에서 찾은 템플릿 수: {len(templates)}")
        # st.write(f"템플릿 목록: {templates}")
        return [doc['name'] for doc in templates]
    st.warning("데이터베이스 연결 없음, 템플릿을 가져올 수 없습니다.")
    return []

def insert_template(db, template_name, template_data):
    if db is not None:
        collection = db['templates']
        template_data['name'] = template_name
        collection.update_one({'name': template_name}, {'$set': template_data}, upsert=True)

def get_template(db, template_name):
    if db is not None:
        collection = db['templates']
        template = collection.find_one({'name': template_name})
        if template:
            # _id 필드 제거 (JSON으로 직렬화할 수 없음)
            template.pop('_id', None)
        return template
    return None

def insert_json_files_to_mongodb(json_directory):
    db = connect_to_mongodb()
    if db is not None:
        for filename in os.listdir(json_directory):
            if filename.endswith('.json'):
                with open(os.path.join(json_directory, filename), 'r') as file:
                    template_data = json.load(file)
                    template_name = os.path.splitext(filename)[0]
                    insert_template(db, template_name, template_data)
        # st.success("모든 JSON 파일이 MongoDB에 삽입되었습니다.")
    else:
        st.error("MongoDB에 연결할 수 없어 JSON 파일을 삽입할 수 없습니다.")

# MongoDB를 사용할 수 없는 경우를 위한 대체 함수
def get_templates_from_files(directory):
    templates = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            templates.append(os.path.splitext(filename)[0])
    return templates