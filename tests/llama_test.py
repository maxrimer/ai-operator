import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"


SYSTEM_PROMPT = """
   Ты — русскоязычный помощник банка.
   Всегда отвечай только на русском, без английских слов.
   Если необходимо использовать иностранный термин, дай русскую транслитерацию.
   """
prompt = "Кто такой Чарльз Бэббидж?"

body = {"model": "llama3:8b-instruct-q4_K_M",
        "prompt": SYSTEM_PROMPT + prompt,
        "stream": False,
        "temperature": 0}


resp = requests.post(OLLAMA_URL, data=json.dumps(body), timeout=30)
text = resp.json()['response']
print(text)









