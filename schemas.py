from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    purpose: str
    question: str
    prompt_name: str = "default_summary"

class IntegratedQueryRequest(BaseModel):
    purposes: List[str]
    question: str
    prompt_name: str = "default_integrated"

class QueryResponse(BaseModel):
    answer: str
