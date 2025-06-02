import os
from openai import OpenAI
import openai
import requests
import time
import json
import time

    
def chat_deepseek_v3(client, system, query):
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ]
        )
        
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

