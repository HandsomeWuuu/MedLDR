from openai import OpenAI  
import re
import requests
import json
import os   


def chat_gemini_2_5(client, system, query):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {'sk-xxx'}"
        }
        payload = {
            "model": "gemini-2.5-pro-preview-03-25",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7
        }

        response = client.post("https://api.cursorai.art/v1/chat/completions", headers=headers, json=payload)

        
        if response.status_code != 200:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

        resp_json = response.json()

        print('response json:', resp_json)
        if "choices" not in resp_json or not resp_json["choices"]:
            raise ValueError("API response does not contain valid choices")

        response_content = resp_json["choices"][0]["message"]["content"]
        token_dict = {
            "prompt_tokens": resp_json["usage"]["prompt_tokens"],
            "completion_tokens": resp_json["usage"]["completion_tokens"],
            "total_tokens": resp_json["usage"]["total_tokens"]
        }

        return response_content, token_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

