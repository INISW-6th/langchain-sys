import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("Hugging Face 로그인 성공")
    except Exception as e:
        print(f"Hugging Face 로그인 실패: {e}")

DATA_PATH = "/content/drive/MyDrive/Textbook-Data/rag적용"

RAG_CONFIG = {
    "chunking": {"method": "custom", "chunk_size": 700, "chunk_overlap": 140},
    "embedding": {"model_type": "huggingface", "model_name": "jhgan/ko-sroberta-multitask"},
    "vector_db": "faiss", "initial_top_k": 20, "rerank_top_k": 5, "reranker": "bge",
    "cohere_api_key": os.getenv("COHERE_API_KEY"), "llm": "exaone", "max_total_docs": 10
}

LLM_MODEL_MAP = { "exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" }
SYSTEM_PROMPT_MAP = { "exaone": "수업자료 생성 도우미" }
