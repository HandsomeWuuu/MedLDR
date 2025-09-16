import requests
import json
import os
from openai import OpenAI

def collect_stream_response(stream) -> dict:
    """
    Collect all result fragments from the streaming call, separating the reasoning process and the final answer.
    """
    reasoning_content = ""  # Complete reasoning process
    answer_content = ""     # Complete response
    
    try:
        # Directly iterate over the stream without type checking
        for chunk in stream:
       
            # print(f"Chunk type: {type(chunk)}")
            
    
            if not hasattr(chunk, 'choices') or not chunk.choices:
                print("Chunk has no choices, skipping...")
                continue
                
            delta = chunk.choices[0].delta
            # print(f"Delta: {delta}")
            
            # Collect reasoning content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
                # print(f"Added reasoning: {delta.reasoning_content[:50]}...")
            
            # Collect reply content
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
                # print(f"Added content: {delta.content[:50]}...")
                
    except Exception as e:
        print(f"Error occurred while processing streaming response: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    
    return {
        "response": answer_content,
        "reasoning": reasoning_content if reasoning_content else None
    }

def chat_qwen_qwen3_235b_a22b_official(client, system, query, think=False, temperature=1.0):
    
    enable_thinking = think

    try:
        use_client = OpenAI(
            api_key="sk-xxx",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = use_client.chat.completions.create(
            model="qwen3-235b-a22b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
            extra_body={"enable_thinking": enable_thinking},
            stream=enable_thinking
        )
        
        print(f'completion type: {type(completion)}')
        
        # Streaming call handling
        if enable_thinking:
            print("Start processing streaming response...")
            stream_result = collect_stream_response(completion)
            result_dict = {
                "response": stream_result["response"],
                "reasoning": stream_result["reasoning"]
            }
            token_dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        else:
            print(f'Non-streaming completion: {completion}')
            # Non-streaming call handling
            result_dict = {
                "response": completion.choices[0].message.content
            }
            token_dict = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }

        return result_dict, token_dict
        
    except Exception as e:
        print(f"API call error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "response": f"API call failed: {str(e)}",
            "reasoning": None
        }, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    


def test_chat_qwen_official():
    client = requests.Session()
    system = "You are a helpful assistant."
    query = "Which is greater, 9.11 or 9.8?"
    
    # Test without thinking mode
    print("\n=== Test without thinking mode ===")
    response_content, token_dict = chat_qwen_qwen3_235b_a22b_official(
        client, 
        system, 
        query, 
        think=False,
        temperature=1.0
    )
    print("Response content:", response_content)
    print("Token statistics:", token_dict)
    
    # Test with thinking mode
    print("\n=== Test with thinking mode ===")
    response_content, token_dict = chat_qwen_qwen3_235b_a22b_official(
        client, 
        system, 
        query, 
        think=True,
        temperature=1.0
    )
    print("Full response content:", response_content)
    print("Token statistics:", token_dict)

if __name__ == "__main__":
    test_chat_qwen_official() 
    # chat_qwen_235b_a22b_think()