from flask import Flask, request, jsonify
from waitress import serve
from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "mistraldb"
embedding_function = get_embedding_function()
model = Ollama(model="mistral")
print("EMBEDDING STARTS")
print(embedding_function)
print("EMBEDDING STOPS")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

app = Flask(__name__)

@app.route('/appinfo', methods=['GET'])
def appinfo():
    return "Hello World! Im still Alive Please reach to Christon James if you not see this message!!!"

@app.route('/getprompt', methods=['GET'])
def getprompt():
    input_query = request.get_json()
    print(input_query)
    question = input_query.get('query')
    limit = input_query.get('limit')

    #embeddings = OllamaEmbeddings(model="mistral")
    #print(type(embeddings))
    #vectorstore_ = FAISS.load_local("FAISS_", embedding_function, "index", allow_dangerous_deserialization=True)
    results = db.similarity_search_with_score(question, k=4)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    if limit != None and int(limit) < len(context_text):
        context_text = context_text[0:limit]
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    Please provide answer in JSON format with tags : question, answer.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    return (prompt)

@app.route('/queryrag', methods=['POST'])
def hello():
    data = request.get_json()
    print(data)
    # Do something with the data
    question = data.get('query')
    print(question)
    response_text = query_rag(question, db)
    return jsonify(response_text)

if __name__ == '__main__':
    # app.run()
    print("serving start")
    serve(app,  port=2501,cleanup_interval=90 )
    print("serving closed")