import torch, numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import Document
from typing import List
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
