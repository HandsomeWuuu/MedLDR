from openai import OpenAI  
import re
import requests
import json
import os   


def chat_o4_mini(client, system, query):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {'sk-xxx'}"
        }
        payload = {
            "model": "o4-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "reasoning_effort":'high'
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


def chat_o4_mini_with_zzz(client, system, query):
    try:
        resp = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            temperature=1,  # 设置温度
            # logprobs=10,
            # max_tokens=50,
            # n=1,
            # logprobs=10,
        )   
        
        # 检查 resp 和 resp.choices 是否为 None
        if resp is None or not resp.choices:
            raise ValueError("API response is None or does not contain choices")
        
        print('o4-mini resp:', resp)
        response = resp.choices[0].message.content
        token_dict = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens
        }

        return response, token_dict
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

