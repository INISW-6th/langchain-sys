# T-prep LangChain System
[**T-prep**](https://github.com/INISW-6th/t-prep)에서 데이터를 RAG를 활용하고 정보의 질을 향상시키고, 각 템플릿에 맞게 결과물을 생성하는 프로젝트입니다.

## 설명
이 프로젝트는 LangChain을 이용해 여러 가지 모델(임베딩, LLM 등)과 벡터DB, 프롬프트 템플릿을 관리하며, 파이프라인의 각 단계에서 가장 적합한 설정을 선택해 결과물을 생성합니다.

해당 프로젝트에서는 [**평가**](https://github.com/INISW-6th/langchain-eval)를 통해 다음 기능을 위해 아래와 같은 기법과 모델을 사용합니다.

| 단계 | 활용 자료 | 프롬프트 기법 | 모델 설정 |
| :-: | :-: | :-: | :-: |
| **내용 요약** | `교과서` `판서` `사료` | `Prompt Chaining` | `ko-sroberta-multitask` `FAISS` `BGE` `EXAONE-3.5-7.8B` |
| **수업자료 생성** | `판서` `사료` `지도서` | `CoT` `Prompt Chaining` | `ko-sroberta-multitask` `FAISS` `BGE` `EXAONE-3.5-7.8B` |
| **기승전결 맥락** | `교과서` `판서` `사료` | `ToT` `Prompt Chaining` | `ko-sroberta-multitask` `FAISS` `BGE` `EXAONE-3.5-7.8B` |
| **시나리오 작성** | `교과서` `판서` `사료` | `ToT` `Prompt Chaining` | `ko-sroberta-multitask` `FAISS` `BGE` `EXAONE-3.5-7.8B`  |
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

# 한국어 역사문서 기반 RAG 실험 시스템

이 프로젝트는 'LangChain + HuggingFace LLM + 한국어 전용 임베딩 모델'을 활용한 'RAG(Retrieval-Augmented Generation)' 기반 질의응답 시스템 실험입니다.
주요 목표는 다양한 모델 조합과 벡터스토어 세팅에 따른 '정확도 및 교육적 유용성' 비교입니다.

## 실험 목적
- 한국어 기반 문서(교과서, 민족대백과 등)를 바탕으로 한 '정확한 역사 서술 생성'
- 다양한 LLM 및 임베딩/벡터 조합에 따라 '응답 품질 변화 분석'
- '리랭커 적용 유무' 및 '벡터스토어 선택'에 따른 결과 비교
- 교육 콘텐츠로 활용 가능한 출력 구성 평가

![](/src/pipeline.png)

## 기본 실험 조건
- 질문: '조선의 건국과정'
- 프롬프트: 고정
- Chunking 방식: 'recursive' 고정
- Fixed Chunk: 사용 안 함 (문장 단위 깨짐 현상 있음)
- Initial Top-K: 20 → Rerank Top-K: 5

## 실험 조건
1. llm, embedding, vector 고정 → reranker만 변경
2. llm, embedding 고정 → vector, reranker 변경
3. llm 고정 → embedding, vector, reranker 변경
4. llm 변경 → 위 1~3번 실험 반복
5. llm 변경 시 런타임 재시작 필요

## LLM-as-a-Judge: GPT-4 평가 기준
### 1차 평가 기준
1. 교과서·지도안·백과사전 등 '모든 문서'를 충실히 참조했는가?
2. '사실 기반'으로 역사를 서술했는가?

### 2차 평가 기준
1. 정확성 : 문서 내 정보만을 기반으로 응답했는가?
2. 문서 참조 충실도 : 다양한 출처를 균형 있게 활용했는가?
3. 구성력 : 기-승-전-결 구조가 논리적으로 구성됐는가?
4. 교육 유용성 : 중학생이 이해하기 쉬운 설명이었는가?
5. 할루시네이션 여부 : 과장/환상/예측 없이 사실만 언급했는가? 

## RAGAS
### 평가 기준
- faithfulness
- answer_relevancy
- context_recall
- context_precision

## 고정 프롬프트
```
당신은 모든 문서에서만 정보를 추출하여 답변해야 합니다.
당신은 모든 문서를 기반으로 역사적 사실만 전달해야하는 역사선생님 입니다.
검색된 모든 문서 내용에 없는 정보는 절대 추측하지 말고 '해당 정보는 제공되지 않았습니다'라고 답하십시오.
주어진 문서에 없는 추측은 '절대 금지'입니다.
답변은 한번만 출력하도록 하십시오.
답변은 완전한 문장으로 답하시오.
반복되는 문장없이 출력하시오.
한 줄에 한 문장이상 넘지 않도록 출력하고 이어서 아랫줄로 넘어가세요.
문서에 복수 해석이 가능한 경우에는 모두 서술하시오.
예: '아마 ~일 것이다' 형태의 표현은 금지.
최대 10문장이내로 답변하세요.
답변의 각 문장은 어떤 문서의 어느 부분에서 왔는지 정확히 출처를 함께 달아줘.
만약 출처가 없으면 그 내용은 빼고, 추론 없이 대답해줘.
모든 문서를 참고하여 '기승전결' 구조로 수업자료를 작성하시오.

조건:
1. '기(起) - 발단', '승(承) - 전개', '전(轉) - 전환', '결(結) - 마무리' 형식으로 구분하여 작성
2. 각 항목은 간결하게 2~3문장
3. 마지막에 '전체 맥락 요약' 항목을 추가
4. 교과서 스타일의 설명체로 작성
```

## 결과
![](/src/eval-judge.png)

![](/src/eval-ragas.png)

![](/src/eval-total.png)


## 이슈
- 'FronyAI/frony-embed-large-ko-v1' 임베딩 모델 실행 안 됨 (2048차원 → GPU 용량 문제 추정)
- 'Qwen 8B' 모델: 한국어로만 답하라고 프롬프트를 줘도 영어로 응답함
