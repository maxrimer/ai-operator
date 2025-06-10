import requests
import json


def call_local_llm(prompt: str, max_tokens: int = 32, temperature: float = 0.0):
    ollama_url = "http://localhost:11434/api/generate"
    body = {"model": "llama3:8b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature}
    resp = requests.post(ollama_url, data=json.dumps(body), timeout=30)
    text = resp.json()['response']
    return text
