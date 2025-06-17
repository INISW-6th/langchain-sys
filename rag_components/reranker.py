import torch, numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import Document
from typing import List
# 1. BgeReranker 클래스: BGE 모델로 쿼리-문서 쌍의 점수를 계산해 상위 문서 재정렬  
# 2. 토크나이저와 시퀀스 분류 모델을 로드하고 GPU에 배치  
# 3. _compute_score 메서드로 쿼리와 문서 간 유사도 점수 산출  
# 4. rerank 메서드로 점수 기반 상위 top_n개 문서 반환  
# 5. CohereReranker 클래스: Cohere API를 사용해 쿼리-문서 재정렬 수행  
# 6. API 키로 클라이언트 생성 후, rerank 메서드로 top_n 문서 재정렬 및 반환  

class BgeReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda").eval()
    def _compute_score(self, query: str, document: str) -> float:
        inputs = self.tokenizer(query, document, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            return self.model(**inputs).logits.item()
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        scores = [self._compute_score(query, doc.page_content) for doc in documents]
        sorted_indices = np.argsort(scores)[::-1]
        return [documents[i] for i in sorted_indices[:top_n]]
class CohereReranker:
    def __init__(self, api_key: str, model_name: str = "rerank-multilingual-v2.0"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model_name = model_name
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        response = self.client.rerank(query=query, documents=[doc.page_content for doc in documents], top_n=top_n, model=self.model_name)
        return [documents[result.index] for result in response.results]
