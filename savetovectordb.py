# from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

pdf_docs = ['data/WAARR.pdf', 'data/Order_Booking_System.pdf']


def get_pdf_text(pdf_docs):
    # print(type(pdf_docs))
    text = ""
    for pdf in pdf_docs:
        # print(pdf)
        pdf_reader = PdfReader(pdf)
        i = len(pdf_reader.pages)
        for page in pdf_reader.pages:
            # print(page.extract_text())
            print(i)
            chucks = get_text_chunks(page.extract_text())
            get_vectorstore(chucks)
            # text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # print(type(chunks))
    # print(chunks)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="mistral", base_url='http://localhost:11434')  # OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    if os.path.exists("FAISS_"):
        print("Subsequent Calls for existing path")
        vectorstore_ = FAISS.load_local("FAISS_", embeddings, allow_dangerous_deserialization=True)
        vectorstore_.merge_from(vectorstore)
        vectorstore_.save_local("FAISS_")
    else:
        print("Creating on New Path")
        vectorstore.save_local("FAISS_")

    return vectorstore


def retrieve_context(input_query):
    embeddings = OllamaEmbeddings(model="mistral",
                                  base_url='https://chatdocsllm-nchristj-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/codeserver/proxy/11434')
    vectorstore_ = FAISS.load_local("FAISS_", embeddings, allow_dangerous_deserialization=True)
    results = vectorstore_.similarity_search_with_score(input_query, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return (context_text)


# raw_text = get_pdf_text(pdf_docs)
print(retrieve_context("Define WAARR"))
