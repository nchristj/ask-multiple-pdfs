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
model = BertModel.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

# FAISS index for storing document embeddings
dimension = 768  # FinBERT output dimension
index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
doc_embeddings = []
doc_metadata = []

# Constants for file paths
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'metadata.pkl'


@app.route('/appinfo', methods=['GET'])
def appinfo():
    return jsonify({"message": "Document stored successfully."}), 201


@app.route('/storetext', methods=['POST'])
def store_document():
    data = request.json
    text = data.get('text', '')

    # Tokenize the input text and get embeddings
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Average pooling

    # Store embedding and metadata
    index.add(embedding)  # Add to FAISS index
    doc_embeddings.append(embedding)  # Store embedding
    doc_metadata.append(text)  # Store text for reference

    return jsonify({"message": "Document stored successfully."}), 201


@app.route('/store', methods=['POST'])
def store_document_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files['file']
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

    # Read the PDF file and extract text
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""

    headers = []
    # Iterate through each page in the document
    chunks = []
    current_chunk = {"header": None, "content": ""}

    # Iterate through each page in the document
    for page_num, page in enumerate(pdf_document, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size = span["size"]  # Font size of the text
                        text = span["text"].strip()

                        # If the font size is larger than a threshold, treat it as a header
                        if font_size > 12 and text:
                            # If there's an existing header, save its chunk
                            if current_chunk["header"]:
                                chunks.append(current_chunk)

                            # Start a new chunk for the new header
                            current_chunk = {"header": text, "content": ""}
                        else:
                            # Append the text to the current header's content
                            current_chunk["content"] += text + " "

    # Add the last chunk if there's remaining content
    if current_chunk["header"]:
        chunks.append(current_chunk)

    result_chunks = []
    # Print the header and its content for each chunk
    for chunk in chunks:
        result_chunks.append(chunk['header'] + "\n" + chunk['content'])
    pdf_document.close()

    # Chunking the text
    #chunk_size = 512  # Characters per chunk
    #chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size) if
    #          full_text[i:i + chunk_size].strip()]
    chunks = result_chunks
    # Initialize embeddings and metadata
    doc_embeddings = []
    doc_metadata = []

    # Load existing embeddings and metadata if the files exist and are not empty
    if os.path.exists(EMBEDDINGS_FILE) and os.path.getsize(EMBEDDINGS_FILE) > 0:
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                doc_embeddings = pickle.load(f)
        except EOFError:
            doc_embeddings = []

    if os.path.exists(METADATA_FILE) and os.path.getsize(METADATA_FILE) > 0:
        try:
            with open(METADATA_FILE, 'rb') as f:
                doc_metadata = pickle.load(f)
        except EOFError:
            doc_metadata = []

    # Store each chunk's embedding
    for chunk in chunks:
        # Tokenize the input text and get embeddings
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Average pooling

        # Add to FAISS index
        index.add(embedding)

        # Append new embeddings and metadata
        doc_embeddings.append(embedding)
        doc_metadata.append(chunk)

    # Save embeddings and metadata to local files
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(doc_embeddings, f)

    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(doc_metadata, f)

    return jsonify({"message": f"{len(chunks)} chunks stored successfully."}), 201


@app.route('/query', methods=['POST'])
def query_documents():
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
    query_text = data.get('text', '')

    # Tokenize the query text and get embedding
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
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
    return jsonify(results)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    serve(app, port=2501, cleanup_interval=90)
