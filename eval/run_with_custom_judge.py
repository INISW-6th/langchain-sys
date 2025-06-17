import os
import gc
import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

# 우리 시스템의 핵심 구성 요소 및 새로 만든 평가기 임포트
from config import DATA_PATH
from rag_pipeline import ModularRAG
from rag_components.loader import load_purpose_docs
from custom_judge_evaluator import evaluate_answer_with_custom_judge, EVAL_PROMPT_TEMPLATE

# ===================================================================
# 1. 실험 변수 정의 (★★★★★ 여기가 핵심 ★★★★★)
# ===================================================================

EXPERIMENT_CONFIGS = [
    {
        "name": "SBERT-FAISS-BGE-Qwen",
        "embedding_model": "jhgan/ko-sroberta-multitask",
        "vector_db": "faiss",
        "reranker": "bge",
        "generation_llm": "Qwen/Qwen1.5-7B-Chat"
    },
    {
        "name": "SBERT-FAISS-NoReranker-Qwen",
        "embedding_model": "jhgan/ko-sroberta-multitask",
        "vector_db": "faiss",
        "reranker": None,
        "generation_llm": "Qwen/Qwen1.5-7B-Chat"
    },
]

FIXED_QUESTION = "조선의 건국 과정에 대해 기승전결 시나리오를 작성해줘."

# ===================================================================
# 2. 메인 실행 로직
# ===================================================================

def main():
    print("커스텀 LLM 평가자를 이용한 전체 실험 프레임워크를 시작합니다...")
    load_dotenv()

    # --- API 키 설정 ---
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if HF_TOKEN:
        try: login(token=HF_TOKEN); print("Hugging Face 로그인 성공.")
        except Exception as e: print(f"Hugging Face 로그인 실패: {e}")
            
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: 커스텀 평가자를 위한 OPENAI_API_KEY가 설정되지 않았습니다.")

    # --- 공통 데이터 로드 ---
    purpose_docs = load_purpose_docs(DATA_PATH)
    all_results = []
    
    # --- 정의된 모든 실험 조합을 순차적으로 실행 ---
    for exp_config in EXPERIMENT_CONFIGS:
        print("\n\n" + "="*70)
        print(f"▶ 실험 시작: {exp_config['name']}")
        print("="*70)
        
        # --- RAG 파이프라인 동적 생성 ---
        current_rag_config = {**exp_config, **{"chunking": {"method": "custom"}, "initial_top_k": 5}}
        rag_pipeline = ModularRAG(current_rag_config, purpose_docs)

        # --- 답변 생성 (통합 검색 방식 사용) ---
        print(f"'{exp_config['name']}' 방식으로 답변 생성 중...")
        purposes_to_use = ['민족대백과', '교과서']
        # 통합 검색용 프롬프트는 {purposes}가 필요 없으므로 EVAL_PROMPT_TEMPLATE 사용 가능
        generated_answer = rag_pipeline.ask_naive_rag(
            purposes=purposes_to_use,
            question=FIXED_QUESTION,
            prompt_template=EVAL_PROMPT_TEMPLATE.format(context="{context}", question="{question}", purposes="{purposes}")
        )
        print("답변 생성 완료.")
        
        # --- 커스텀 LLM 평가자에게 채점 요청 ---
        print("생성된 답변을 커스텀 LLM 평가자에게 보내 채점을 시작합니다...")
        score, reason = evaluate_answer_with_custom_judge(generated_answer)
        print(f"채점 완료: {score}점")

        # --- 결과 기록 ---
        all_results.append({
            "experiment_name": exp_config['name'],
            "generated_answer": generated_answer,
            "judge_score": score,
            "judge_reason": reason
        })

        # --- GPU 메모리 정리 ---
        print(f"\n--- '{exp_config['name']}' 실험 완료. GPU 캐시 정리 ---")
        del rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    # --- 최종 결과 종합 및 저장 ---
    if all_results:
        final_report = pd.DataFrame(all_results)
        print("\n\n" + "="*70)
        print("✅ 모든 실험 완료. 최종 종합 보고서:")
        print("="*70)
        print(final_report[['experiment_name', 'judge_score', 'judge_reason']])
        
        report_path = "custom_judge_report.csv"
        final_report.to_csv(report_path, index=False, encoding='cp949')
        print(f"\n최종 보고서가 '{report_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
