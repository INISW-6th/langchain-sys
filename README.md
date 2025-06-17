# T-prep LangChain System
[**T-prep**](https://github.com/INISW-6th/t-prep)에서 데이터를 RAG를 활용하고 정보의 질을 향상시키고, 각 템플릿에 맞게 결과물을 생성하는 프로젝트입니다.

## 설명
이 프로젝트는 LangChain을 이용해 여러 가지 모델(임베딩, LLM 등)과 벡터DB, 프롬프트 템플릿을 관리하며, 파이프라인의 각 단계에서 가장 적합한 설정을 선택해 결과물을 생성합니다.

해당 프로젝트에서는 [**평가**](https://github.com/INISW-6th/langchain-eval)를 통해 다음 기능을 위해 아래와 같은 기법과 모델을 사용합니다.

| 단계 | 활용 자료 | 프롬프트 기법 | 모델 설정 | 내용 |
| :-: | :-: | :-: | :-: | :-: |
| **수업자료 생성** | `교과서` `판서` `사료` `지도서` | `CoT` `Prompt Chaining` | `ko-sroberta-multitask` `FAISS` `EXAONE-3.5-7.8B` | - |
| **내용 요약** | `교과서` `판서` `사료` | `Prompt Chaining` | `KoSimCSE-roberta` `ChromaDB` `BGE` `EXAONE-3.5-7.8B` | - |
| **기승전결 맥락** | `교과서` `판서` `사료` | `ToT` `Prompt Chaining` | `KoSimCSE-roberta` `ChromaDB` `BGE` `EXAONE-3.5-7.8B` | - |
| **시나리오 작성** | `교과서` `판서` `사료` | `ToT` `Prompt Chaining` | `KoSimCSE-roberta` `ChromaDB` `EXAONE-3.5-7.8B` | - |
| **삽화 생성** | `교과서` | `Prompt with Constraints` | `DALLE3` | - |


## 프로젝트 구조
```
ipynb (Colab 노트북)
├── server.py       # API 서버 (FastAPI + ngrok)
├── worker.py       # LLM 워커 (RAG 처리)
├── config.py       # 설정값 관리
├── ModularRAGExperiment.py # RAG 핵심 로직
├── prompts/        # 프롬프트 템플릿 저장소
├── queue.json      # 처리 대기 중인 질문 목록
└── answers.json    # 처리 완료된 답변 목록
```

