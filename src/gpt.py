
import openai
import os
import json
from tqdm import tqdm
import argparse
import time
import random
import math

def send_to_chat_gpt(
    model: str, messages: str, max_tokens: int = 4096, temperature: float = 0.0
) -> str:
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature
    )
    response_content = response.choices[0].message.content
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_prompt(model, prompt, *args, **kwargs):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    response = send_to_chat_gpt(model,messages, *args, **kwargs)
    return response


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--option', type=str, help='''overall, 
                                                        noCoT, 
                                                        noCaption, 
                                                        noASR, 
                                                        noICL, 
                                                        noICLCoT''', default='overall')
    parser.add_argument('--prompt_path', type=str, help='', default='')
    parser.add_argument('--log_path', type=str, help='', default='outputs')
    args = parser.parse_args()

    openai.api_key = os.environ['OPENAI_API_KEY']
    model = "gpt-3.5-turbo-16k"

    log_path = os.path.join(args.log_path,args.option+'_'+str(time.time())+'.json')
    log = []

    with open(args.prompt_path,'r') as file:
        prompts = json.load(file)

    # prompts -> list([str,])
    for prompt in tqdm(prompts):
        try:
            res = send_prompt(model, prompt, temperature = 0.0)
            answer = res['content']
            log.append({'response':res,'prompt':prompt,'error':''})
        except Exception as e:
            print('Error!\t',e)
            log.append({'response':None,'prompt':prompt,'error':str(e)})
    
    with open(log_path,'w') as file:
        json.dump(log,file)
            