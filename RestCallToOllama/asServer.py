from flask import Flask, request, jsonify
from waitress import serve
from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import string
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import numpy as np
import faiss  # Make sure you have the faiss-cpu version installed
import torch
from waitress import serve
import fitz
import json
import os
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load FinBERT model and tokenizer
model_name = 'yiyanghkust/finbert-tone'
tokenizer = BertTokenizer.from_pretrained(model_name)
modelBert = BertModel.from_pretrained(model_name)
modelBert.eval()  # Set model to evaluation mode

# FAISS index for storing document embeddings
dimension = 768  # FinBERT output dimension
index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
doc_embeddings = []
doc_metadata = []

# Constants for file paths
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'metadata.pkl'

CHROMA_PATH = "mistral_fr"
embedding_function = get_embedding_function()

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

app = Flask(__name__)


@app.route('/appinfo', methods=['GET'])
def appinfo():
    return "Hello World! Im still Alive Please reach to Christon James if you not see this message!!!"


@app.route('/getpromptold', methods=['GET'])
def getpromptold():
    input_query = request.get_json()
    print(input_query)
    question = input_query.get('query')
    limit = input_query.get('limit')

    # embeddings = OllamaEmbeddings(model="mistral")
    # print(type(embeddings))
    # vectorstore_ = FAISS.load_local("FAISS_", embedding_function, "index", allow_dangerous_deserialization=True)
    # results = db.similarity_search_with_score(question, k=4)
    # embeddings = OllamaEmbeddings(model="mistral")
    # vectordb = FAISS.load_local(folder_path="FAISS_",embeddings=embeddings,allow_dangerous_deserialization=True)
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # if limit != None and int(limit) < len(context_text):
    #    context_text = context_text[0:limit]
    vector = embedding_function.embed_query(question)
    results = db.similarity_search_by_vector_with_relevance_scores(vector, k=5)
    score_max = 0
    context_text = ""
    threash_hold = 0.7
    tokens = question.split(" ")
    for doc, _score in results:
        if score_max < _score:
            print(_score)
            context_text = doc.page_content
            score_max = _score
    print(len(context_text))
    print(context_text)
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    Please provide answer in JSON format with tag : answer.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    print(prompt)
    return (prompt)


@app.route('/getprompt', methods=['GET'])
def getprompt():
    # Load existing embeddings and metadata from files
    doc_embeddings = []
    doc_metadata = []

    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            doc_embeddings = pickle.load(f)

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'rb') as f:
            doc_metadata = pickle.load(f)

    # Ensure the FAISS index is populated from loaded embeddings
    index.reset()  # Reset the FAISS index
    if doc_embeddings:
        index.add(np.vstack(doc_embeddings))  # Add all embeddings back to the index

    # Continue with the query process
    data = request.json
    query_text = data.get('query', '')

    # Tokenize the query text and get embedding
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = modelBert(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).numpy()

    # Perform similarity search
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    # Prepare the response
    results = []
    for i in range(k):
        if distances[0][i] < float('inf'):
            results.append({
                "text": doc_metadata[indices[0][i]],
                "distance": distances[0][i]
            })
    # Convert float32 to native float in the list of dictionaries
    for item in results:
        for key, value in item.items():
            if isinstance(value, np.float32):
                item[key] = float(value)
                print(float(value))

    score_max = 0
    context_text_first = ""
    print("length for results" + str(len(results)))
    for item in results:
        if 'distance' in item and 'text' in item:
            if score_max < item['distance']:
                context_text = item['text']
                if score_max == 0:
                    context_text_first = item['text']
                score_max = item['distance']

    print(len(context_text_first + context_text))

    PROMPT_TEMPLATE = """
    Answer the question based only the two contexts results, where
    context1: {context1},

    context2: {context2},

    ---

    Answer the question based on the above context: {question}
    Please provide answer in JSON format with tag : answer.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context1=context_text_first, context2=context_text, question=query_text)
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
    print("make model up with cache starts")
    serve(app, port=2501, cleanup_interval=90)
    print("serving closed")