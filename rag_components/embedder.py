from langchain_community.embeddings import HuggingFaceEmbeddings
# 1. 설정(config)에서 임베딩 모델 타입과 이름 추출  
# 2. 모델 타입이 'huggingface'인 경우 HuggingFaceEmbeddings 로드  
# 3. CUDA 디바이스와 normalize 설정 적용  
# 4. 지원하지 않는 모델 타입인 경우 예외 발생 
def get_embedding_model(config: dict):
    model_type = config["embedding"]["model_type"]
    model_name = config["embedding"]["model_name"]
    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
    raise ValueError(f"지원하지 않는 임베딩 타입: {model_type}")
