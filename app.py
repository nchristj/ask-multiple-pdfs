import streamlit as st
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
pdf_docs = ['data/WAARR.pdf', 'data/Order_Booking_System.pdf']
def get_pdf_text(pdf_docs):
    #print(type(pdf_docs))
    text = ""
    for pdf in pdf_docs:
        #print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            print(page.extract_text())
            chucks = get_text_chunks(page.extract_text())
            get_vectorstore(chucks)
            #text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(type(chunks))
    print(chunks)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="mistral") #OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("FAISS")
    return vectorstore


def get_conversation_chain(vectordb):
    llm = Ollama(model="mistral")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main(vectordb):
    #load_dotenv()
    st.set_page_config(page_title="Wealth Tech Insight",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.session_state.conversation = get_conversation_chain(vectordb)
    st.header("Explore Wealth Tech in Citi")
    user_question = st.text_input("Have a doubt, Just Ask Me?:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    #raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
    #text_chunks = get_text_chunks(raw_text)

    # create vector store
    #vectorstore = get_vectorstore(text_chunks)#, option)

    # create conversation chain
    embeddings = OllamaEmbeddings(model="mistral")
    vectordb = FAISS.load_local(folder_path="FAISS_",embeddings=embeddings,allow_dangerous_deserialization=True)
    main(vectordb)
    model = FAISS.from_embeddings()