import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# 1. 주어진 모델 이름으로 토크나이저와 모델 로드 (trust_remote_code=True)
# 2. float16 타입으로 모델을 자동 디바이스에 로드
# 3. text-generation 파이프라인 구성 (샘플링 비활성화, 최대 토큰 제한)
# 4. system_prompt가 있으면 chat 템플릿 적용하여 전체 프롬프트 구성
# 5. 생성된 결과에서 프롬프트 부분 제거 후 응답 텍스트 반환
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
