import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
def get_hf_llm(model_name: str, system_prompt: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, do_sample=False)
    def llm_func(prompt):
        full_prompt = prompt
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            try:
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                full_prompt = f"{system_prompt}\\n\\nUser: {prompt}\\nAssistant:"
        result = pipe(full_prompt)[0]["generated_text"]
        if result.startswith(full_prompt):
            return result[len(full_prompt):].strip()
        return result.strip()
    return llm_func
