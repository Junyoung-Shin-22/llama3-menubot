# from huggingface_hub import login; login() # comment out if already logged in.

import warnings; warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MAX_TURN = 5
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

TOK = AutoTokenizer.from_pretrained(MODEL_ID)
MOD = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

with open('./system.txt', 'rt', encoding='utf-8') as f:
    content = f.read()
    SYSTEM = {"role": "system", "content": content}

terminators = [
    TOK.eos_token_id,
    TOK.convert_tokens_to_ids("<|eot_id|>")
]

def llama_single_inference(messages):
    # assert len(messages) <= MAX_TURN, "conversation is too long"

    ids = TOK.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(MOD.device)
    outputs = MOD.generate(
        ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=0,
        do_sample=True,
        temperature=1.5, # here, using higher temperature to get more randomized result
        top_p=0.9,
    )

    response = outputs[0][ids.shape[-1]:]
    response = TOK.decode(response, skip_special_tokens=True)

    return response

if __name__ == '__main__':
    print('model device:', MOD.device)
    message = [
        SYSTEM,
        {'role': 'user', 'content': 'ㅈㅁㅁ'}
    ]
    print('user:', message[-1]['content'])
    print('bot:', llama_single_inference(message))