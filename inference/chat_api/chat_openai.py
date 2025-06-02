from openai import OpenAI  
import re
import requests

def chat_gpt4o(client, system, query):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            # logprobs=10,
            # max_tokens=50,
            # n=1,
            # logprobs=10,
        )   
        
        # 检查 resp 和 resp.choices 是否为 None
        if resp is None or not resp.choices:
            raise ValueError("API response is None or does not contain choices")
        
        # print('gpt-4o resp:', resp)
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

def chat_o3_mini(client, system, query):
    try:
        resp = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            reasoning_effort='high', # low, medium, high
        )
        
        # print('o3-mini resp:', resp)
        # 检查 resp 和 resp.choices 是否为 None
        if resp is None or not resp.choices:
            raise ValueError("API response is None or does not contain choices")
        
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

def chat_o3_mini_with_cursorai(client, system, query):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {'sk-evS9Nelt6ja1bd3a7DFjR1xcPu5qRgK41K8U1h5bs46MI5Mx'}"
        }
        payload = {
            "model": "o3-mini-high-all",
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
