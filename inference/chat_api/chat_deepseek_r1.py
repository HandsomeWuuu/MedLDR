import requests
from openai import OpenAI
import os

def chat_deepseek_r1_zzz(client, system, query):
    try:
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            # temperature=0.7,  # 设置温度
            # max_tokens=150    # 限制最大输出长度
        )
        print('resp:',resp)
        # 检查 resp 和 resp.choices 是否为 None
        if resp is None or not resp.choices:
            raise ValueError("API response is None or does not contain choices")
        
        # print('resp: ',resp)
        result_dict = {}
        result_dict['answer'] = resp.choices[0].message.content
        result_dict['reasoning'] = resp.choices[0].message.reasoning_content
    
        token_dict = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "reasoning_tokens": resp.usage.completion_tokens_details.reasoning_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens
        }

        return result_dict, token_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None




if __name__ == "__main__":
    system_message = "You are a helpful assistant."
    user_query = "What is the capital of France?"

    test_company = "zzz" 

    
    if test_company == "zzz":
        # 准备服务器
        API_SECRET_KEY = "sk-xxx"
        BASE_URL = "https://api.zhizengzeng.com/v1"
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

        print("*** Response from chat_deepseek_r1_zzz: *** \n")
        response, token_usage = chat_deepseek_r1_zzz(client, system_message, user_query)
        
        print("Response from chat_deepseek_r1_zzz:")
        print(response)
        print("Token usage:")
        print(token_usage)

        