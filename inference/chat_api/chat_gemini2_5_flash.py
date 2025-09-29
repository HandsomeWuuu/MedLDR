from openai import OpenAI  
import re
import requests
import json
import os   


def chat_gemini_2_5_flash(client, system, query,think=False,temperature=1):
    if think:
        print('think is True')
        model_name = "gemini-2.5-flash-thinking"
    else:
        print('think is False')
        model_name = "gemini-2.5-flash-nothinking"
        

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

        # print('response json:', resp_json)

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
            token_dict['reasoning_tokens'] = resp_json["usage"]["completion_tokens_details"]["reasoning_tokens"]
            reasoning_content = resp_json["choices"][0]["message"]["reasoning_content"]
            result_dict['reasoning'] = reasoning_content
        

        return result_dict, token_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def test_chat_gemini_2_5_flash():
    client = requests.Session()
    system = "You are a helpful assistant."
    query = "Which is greater, 9.11 or 9.8?"

    # set your API key to environment variable or directly assign it here
    import os
    os.environ["API_KEY"] = "sk-xxxx"  # replace with your actual API key
    
    response_content, token_dict = chat_gemini_2_5_flash(client, system, query,think=True,temperature=1)
    # response_content, token_dict = chat_gemini_2_5_flash(client, system, query,think=False,temperature=1)
    
    print("Response Content:", response_content)
    print("Token Dictionary:", token_dict)
    

    
if __name__ == "__main__":
    test_chat_gemini_2_5_flash()
