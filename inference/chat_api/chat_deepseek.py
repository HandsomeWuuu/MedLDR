import os
from openai import OpenAI
import openai
import requests
import time
import json
import time

def chat_deepseek_cursorai(client, system, query, think, temperature=1.0):

    if not think:
        model_name = "deepseek-v3-250324"
    else:
        model_name = "deepseek-r1-250528"

    try:
        # Try get the API key from environment variable
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable not set.")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            "temperature": temperature,

        }


        response = client.post("https://api.cursorai.art/v1/chat/completions", headers=headers, json=payload)

        
        if response.status_code != 200:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

        resp_json = response.json()

        print('deepseek response json:', resp_json)

        if "choices" not in resp_json or not resp_json["choices"]:
            raise ValueError("API response does not contain valid choices")

        result_dict = {
            "response": resp_json["choices"][0]["message"]["content"],
        }
        

        token_dict = {
            "prompt_tokens": resp_json["usage"]["prompt_tokens"],
            "completion_tokens": resp_json["usage"]["completion_tokens"],
            "total_tokens": resp_json["usage"]["total_tokens"]
        }

        if think:
            token_dict['reasoning_tokens'] = 0
        
        

        return result_dict, token_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    

if __name__ == "__main__":
    
    system_message = "You are a helpful assistant."
    user_query = "What is the capital of France?"
    
    import requests
    client = requests.Session()

    # set your API key to environment variable or directly assign it here
    import os
    os.environ["API_KEY"] = "sk-xxxx"  # replace with your actual API key
    
    print("Testing chat_deepseek_cursorai think=True")
    response, token_usage = chat_deepseek_cursorai(client, system_message, user_query, think=True)
    # print("Testing chat_deepseek_cursorai think=False") 
    # response, token_usage = chat_deepseek_cursorai(client, system_message, user_query, think=False)

    print("Response from chat_deepseek_cursorai:")
    print(response)
    print("Token usage:")
    print(token_usage)