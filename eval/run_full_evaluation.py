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
from ragas_evaluator import run_ragas_evaluation

# ===================================================================
# 1. 실험 변수 정의
# ===================================================================

# 테스트하고 싶은 RAG 컴포넌트 조합을 리스트에 추가하세요.
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
        "reranker": None, # 리랭커 제외
        "generation_llm": "Qwen/Qwen1.5-7B-Chat"
    },
    # 여기에 다른 임베딩 모델, 벡터DB, LLM 조합을 추가하여 실험 가능
]

# 모든 실험에 동일하게 사용할 고정 질문
FIXED_QUESTION = "위화도 회군 사건의 배경과 이성계의 역할에 대해 설명해줘."

# ===================================================================
# 2. 메인 실행 로직
# ===================================================================

def main():
    print("RAGAS를 이용한 전체 실험 프레임워크를 시작합니다...")
    load_dotenv()

    # --- API 키 설정 ---
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
            print("Hugging Face 로그인 성공.")
        except Exception as e:
            print(f"Hugging Face 로그인 실패: {e}")
            
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: RAGAS 평가를 위한 OPENAI_API_KEY가 설정되지 않았습니다.")
        # return # 키가 없으면 실행 중단

    # --- 공통 데이터 로드 ---
    print("공통 데이터를 로드합니다...")
    purpose_docs = load_purpose_docs(DATA_PATH)
    print("공통 데이터 로딩 완료.")

    all_results = []
    
    # --- 정의된 모든 실험 조합을 순차적으로 실행 ---
    for exp_config in EXPERIMENT_CONFIGS:
        print("\n\n" + "="*70)
        print(f"▶ 실험 시작: {exp_config['name']}")
        print("="*70)
        
        # --- 현재 실험 설정에 맞는 RAG 파이프라인 동적 생성 ---
        # RAG_CONFIG를 현재 실험의 설정으로 만듭니다.
        current_rag_config = {
            "embedding": {"model_type": "huggingface", "model_name": exp_config["embedding_model"]},
            "vector_db": exp_config["vector_db"],
            "reranker": exp_config["reranker"],
            "llm": exp_config["generation_llm"],
            # 기타 고정 설정
            "chunking": {"method": "custom", "chunk_size": 700, "chunk_overlap": 140},
            "initial_top_k": 20, "rerank_top_k": 5, "max_total_docs": 10
        }
        
        rag_pipeline = ModularRAG(current_rag_config, purpose_docs)

        # --- RAGAS 평가 실행 ---
        # 1. Modular (민족대백과) 평가
        df_mod_1 = run_ragas_evaluation(rag_pipeline, FIXED_QUESTION, 'modular', ['민족대백과'])
        df_mod_1['experiment_name'] = exp_config['name']
        df_mod_1['eval_type'] = 'modular_민족대백과'
        all_results.append(df_mod_1)

        # 2. Modular (교과서) 평가
        df_mod_2 = run_ragas_evaluation(rag_pipeline, FIXED_QUESTION, 'modular', ['교과서'])
        df_mod_2['experiment_name'] = exp_config['name']
        df_mod_2['eval_type'] = 'modular_교과서'
        all_results.append(df_mod_2)
        
        # 3. Combined (통합) 평가
        df_comb = run_ragas_evaluation(rag_pipeline, FIXED_QUESTION, 'combined', ['민족대백과', '교과서'])
        df_comb['experiment_name'] = exp_config['name']
        df_comb['eval_type'] = 'combined'
        all_results.append(df_comb)

        # --- GPU 메모리 정리 ---
        print(f"\n--- '{exp_config['name']}' 실험 완료. GPU 캐시 정리 ---")
        del rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    # --- 최종 결과 종합 및 저장 ---
    if all_results:
        final_report = pd.concat(all_results, ignore_index=True)
        print("\n\n" + "="*70)
        print("✅ 모든 실험 완료. 최종 종합 보고서:")
        print("="*70)
        print(final_report)
        
        report_path = "ragas_full_evaluation_report.csv"
        final_report.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\n최종 보고서가 '{report_path}' 파일로 저장되었습니다.")
    else:
        print("수행된 실험이 없습니다.")

if __name__ == "__main__":
    main()
