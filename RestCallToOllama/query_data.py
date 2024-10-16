import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import requests
import json
import time

CHROMA_PATH = "mistraldb"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def generateRestCall(prompt):
    url = "http://localhost:11434/api/generate"

    data = {}
    data['model'] = 'mistral'
    data['prompt'] = prompt
    json_data = json.dumps(data)

    payload = json_data  # "{\r\n  \"model\": \"gemma2\",\r\n  \"prompt\":\"+prompt+\"\r\n }"
    headers = {
        'Content-Type': 'text/plain'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = json.dumps(response.text)

    # print(response.json)
    return response


def query_rag(query_text: str, db):
    start = time.time()
    # Prepare the DB.
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # print(embedding_function)
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    end = time.time()
    print("Time Taken For DB Fetch" + str(end - start))

    start = time.time()
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    list_of_tokens = context_text.split(" ")
    print(len(list_of_tokens))
    prompt = prompt_template.format(context=context_text, question=query_text)

    end = time.time()
    print("Time Taken For Prompt Creation" + str(end - start))

    start = time.time()
    response_text = generateRestCall(prompt)
    end = time.time()
    print("Time Taken For Rest call to generate response" + str(end - start))

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
