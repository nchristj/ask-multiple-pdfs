from flask import Flask, request, jsonify
from waitress import serve
from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

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