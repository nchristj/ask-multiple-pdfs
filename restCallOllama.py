import requests
import json
import base64
import ollama
from PIL import Image
from io import BytesIO
ollama.generate
def encode_image(image_path):
    encoded_string = ""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        return image_data
    return encoded_string

value = base64.b64encode(encode_image('Image_1_test.jpg'))

# Example usage
image_path = "Image_1_test.jpg"  # Replace with the actual path to your image
def getprompt(cookievalue, modelname, query):
    url = "https://chatdocsllm-nchristj-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/codeserver/proxy/2501/getprompt"

    payload = json.dumps({
    "model": modelname,
    "query": query,
    "limit" : 500
    })
    headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,bg;q=0.7,ta;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': cookievalue,
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Content-Type': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.text

def getresponse(cookievalue, modelname, prompt):
    url = "https://chatdocsllm-nchristj-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/codeserver/proxy/11434/api/generate"

    payload = json.dumps({
    "model": modelname,
    "prompt": prompt,
    "format": "json",
    "stream": True
    })
    headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,bg;q=0.7,ta;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': cookievalue,
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        json_data = json.loads(response.text)
        if 'response' in json_data:
            json_answer = json.loads(json_data['response'])
            if 'answer' in json_answer:
                return json_answer['answer']
    return response

def getimagedetails(cookievalue, modelname, prompt):
    url = "https://ollava-nithin5392-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/codeserver/proxy/11434/api/generate"

    payload = json.dumps({
    "model": modelname,
    "prompt": prompt,
    "images" : [value.decode("utf-8")],
        "stream": True
    })
    headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,bg;q=0.7,ta;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': cookievalue,
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, timeout=180)
    if response.status_code == 200:
        print(response.text)
        json_data = json.loads(response.text)
        if 'response' in json_data:
            json_answer = json.loads(json_data['response'])
            if 'answer' in json_answer:
                return json_answer['answer']

    return response

def getresponseembed(cookievalue, modelname, prompt):
    url = "https://chatdocsllm-nchristj-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/codeserver/proxy/11434/api/generate"

    payload = json.dumps({
    "model": modelname,
    "prompt": prompt,
    "format": "json",
    "stream": False
    })
    headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,bg;q=0.7,ta;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': cookievalue,
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        json_data = json.loads(response.text)
        if 'response' in json_data:
            json_answer = json.loads(json_data['response'])
            if 'answer' in json_answer:
                return json_answer['answer']
    return response

modelname = "mistral"
cookievalue = 'ajs_user_id=c4b031fe6d2e6572bf2330361aa19191bc618659; ajs_anonymous_id=6bdb61a9-9951-40cd-b3e4-8d26a586f707; _oauth_proxy=bml0aGluNTM5MkBjbHVzdGVyLmxvY2Fs|1729176929|y2JqdtDqVigQ3pNquiO7JGTwLxA=; analytics_session_id=1729220971971; analytics_session_id.last_access=1729220995255; d0fea3d4bb151171f6c24b60e94cd24c=f7f26983860ff8dff89e1fc12b6879ce'

def getrestcallsdone(question):
    prompt = getprompt(cookievalue,modelname, question)
    response = getresponse(cookievalue, modelname, prompt)
    return response
print(getimagedetails(cookievalue, 'llava', 'What is in this picture?'))
