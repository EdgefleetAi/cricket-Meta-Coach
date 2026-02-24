# llm/generator.py

import torch
from llm.qwen_loader import tokenizer, model

def qwen_generate(prompt, max_new_tokens=300):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )


    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()

    return decoded
