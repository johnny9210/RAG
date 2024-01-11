import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

import streamlit.components.v1 as components

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun

openai_api_key = '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'

def main():
    # st.set_page_config(
    # page_title="TigrisChat",
    # page_icon=":books:")
    
    openai_api_key = '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'

    st.title("_Document :red[QA Chat]_ :tiger:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
# None으로 하는 이유는 -> 해당 내용이 이후에 없기에 처음 명명
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
# None으로 하는 이유는 -> 해당 내용이 이후에 없기에 처음 명명
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
# None으로 하는 이유는 -> 해당 내용이 이후에 없기에 처음 명명
    tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
    with tab1:
        
    # with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file :tiger:",type=['pdf','docx','csv','pptx'],accept_multiple_files=True)
        openai_api_key = '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'
        # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        # if not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")
        #     st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
# 초기값으로 message를 입력
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']
                
                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    # st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    # st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    
    
# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        
        st.session_state.conversation = None
        st.session_state.chat_history = None
        st.session_state.processComplete = None
        # search = st.text_input("What do you want to search for?")
        # components.iframe(f"https://www.google.com/search?igu=1&ei=&q={search}", height=1000)
        st.title("🔎 LangChain - Chat with search")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
            search = DuckDuckGoSearchRun(name="Search")
            search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용, 빈파일 생성 후 해당 파일명을 지정하는 부분
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue()) # 파일 불러온 후 해당 파일을 file_name으로 생성한 파일에 덮어쓰는 부분
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        # elif '.docx' in doc.name:
        #     loader = Docx2txtLoader(file_name)
        #     documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        elif '.csv' in doc.name:
            loader = CSVLoader(file_name)
            documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key='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', model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
