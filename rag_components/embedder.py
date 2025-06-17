from langchain_community.embeddings import HuggingFaceEmbeddings
def get_embedding_model(config: dict):
    model_type = config["embedding"]["model_type"]
    model_name = config["embedding"]["model_name"]
    if model_type == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
    raise ValueError(f"지원하지 않는 임베딩 타입: {model_type}")
