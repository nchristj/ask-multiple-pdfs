import requests
import json

def getprompt(cookievalue, modelname, query):
    url = "https://chatdocsllm-nchristj-dev.apps.sandbox-m2.ll9k.p1.openshiftapps.com/codeserver/proxy/2501/getprompt"

    payload = json.dumps({
    "model": modelname,
    "query": query
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
cookievalue = 'ajs_user_id=5f8e64255d4b1b169dda6eb7ccb8ba879c593fe0; ajs_anonymous_id=09c0e1a7-8d45-4024-946c-ad0f7f3a9340; _oauth_proxy=bmNocmlzdGpAY2x1c3Rlci5sb2NhbA==|1729044885|TjsXsLsySIP2uAAiuaxRK41G0hM=; _streamlit_xsrf=2|beab9f4f|58e663cf0be897a29b016e04cdfaa521|1729045055; analytics_session_id=1729077339662; analytics_session_id.last_access=1729077353792; 0a29b96fbcf1b5b4496eec99edbd0f75=65d59b6452a90b54a1de4b050ce9174e'

def getrestcallsdone(question):
    prompt = getprompt(cookievalue,modelname, question)
    response = getresponse(cookievalue, modelname, prompt)
    return response
