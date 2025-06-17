import os
import re
import pandas as pd
from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# RAG 파이프라인 클래스는 외부에서 인자로 전달받습니다.
# from rag_pipeline import ModularRAG

# 평가에 사용할 프롬프트는 이 파일 내에서 관리합니다.
EVAL_PROMPT_TEMPLATE = """
당신은 중학생에게 역사를 창의적으로 설명해주는 AI 도우미입니다.
다음 지침에 따라 기승전결 구조로 시나리오를 생성하세요.
---
[지문]
{context}
[질문]
{question}
[시나리오 시작]
기:
"""

def run_ragas_evaluation(rag_pipeline, question: str, eval_type: str, purposes: List[str]) -> pd.DataFrame:
    """
    주어진 RAG 파이프라인과 질문으로 답변을 생성하고 RAGAS로 평가합니다.
    
    Args:
        rag_pipeline: 평가할 ModularRAG 클래스의 인스턴스.
        question: 평가에 사용할 질문.
        eval_type: 'modular' 또는 'combined'.
        purposes: 사용할 purpose 리스트.
        
    Returns:
        평가 결과가 담긴 pandas DataFrame.
    """
    print(f"\n===== RAGAS 평가 시작: Type='{eval_type}', Purposes='{purposes}', Question='{question}' =====")
    
    # 1. RAG 답변 생성
    if eval_type == 'modular':
        purpose = purposes[0]
        # invoke를 사용하여 문서 검색
        docs = rag_pipeline.rag_instances[purpose]["retriever"].invoke(question)
        context_text = "\n\n---\n\n".join(doc.page_content.strip() for doc in docs[:2])
        llm_instance = rag_pipeline.rag_instances[purpose]["llm"]
        prompt_template = EVAL_PROMPT_TEMPLATE
        
        filled_prompt = prompt_template.format(context=context_text, question=question)
        answer = llm_instance(filled_prompt)
        
    elif eval_type == 'combined':
        all_contexts = []
        for p in purposes:
            docs = rag_pipeline.rag_instances[p]["retriever"].invoke(question)
            all_contexts.append(docs[0].page_content.strip() if docs else "")
        context_text = "\n\n---\n\n".join(all_contexts)
        llm_instance = rag_pipeline.rag_instances[purposes[0]]["llm"]
        prompt_template = EVAL_PROMPT_TEMPLATE

        filled_prompt = prompt_template.format(context=context_text, question=question)
        answer = llm_instance(filled_prompt)
    else:
        raise ValueError("eval_type은 'modular' 또는 'combined' 여야 합니다.")

    # 2. RAGAS 데이터셋 생성
    dataset_dict = {
        "question": [question],
        "answer": [answer.strip()],
        "contexts": [[context_text]],
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # 3. RAGAS 평가 실행
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    result = evaluate(dataset=dataset, metrics=metrics)
    
    df = result.to_pandas()
    print("평가 완료. 점수:")
    print(df)
    return df
