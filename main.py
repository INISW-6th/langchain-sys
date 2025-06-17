import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pyngrok import ngrok, conf
import os
import json
import gc
import torch
import traceback # 에러 상세 출력을 위한 라이브러리

# 시스템 구성 요소 임포트
from schemas import QueryRequest, IntegratedQueryRequest, QueryResponse
from config import RAG_CONFIG, DATA_PATH
from rag_pipeline import ModularRAG
from rag_components.loader import load_purpose_docs
NGROK_STATIC_DOMAIN = "man-touching-malamute.ngrok-free.app"
# --- ngrok 설정 ---
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTHTOKEN")
if NGROK_AUTH_TOKEN:
    conf.get_default().auth_token = NGROK_AUTH_TOKEN

app = FastAPI()

# --- CORS 설정 ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 백그라운드 작업을 위한 함수 정의 ---
def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    print("--- 백그라운드 GPU 캐시 정리 완료 ---")

# --- 서버 시작 로직 ---
@app.on_event("startup")
def startup_event():
    # ... (이전과 동일)
    print("서버 시작... RAG 파이프라인 및 프롬프트를 초기화합니다.")
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            app.state.prompts = json.load(f)
        print("프롬프트 파일 로딩 완료.")
    except FileNotFoundError:
        print("경고: prompts.json 파일을 찾을 수 없습니다.")
        app.state.prompts = {}
    purpose_docs = load_purpose_docs(DATA_PATH)
    app.state.rag_pipeline = ModularRAG(RAG_CONFIG, purpose_docs)
    print("RAG 파이프라인 준비 완료.")

# --- API 엔드포인트 정의 ---
@app.get("/")
def read_root():
    return {"message": "Modular RAG API is running."}

@app.post("/ask", response_model=QueryResponse)
def ask(request: Request, query: QueryRequest, background_tasks: BackgroundTasks):

    # ===================================================================
    # 디버깅을 위한 로그 추가
    # ===================================================================
    print("\n\n✅ [1/5] /ask 엔드포인트 요청 수신")
    print(f"  - 수신된 Body: {query.dict()}")

    rag_pipeline = request.app.state.rag_pipeline

    print("✅ [2/5] 프롬프트 조회 시작...")
    prompt_template = request.app.state.prompts.get(query.prompt_name)
    if not prompt_template:
        print(f"❌ ERROR: '{query.prompt_name}' 프롬프트를 찾을 수 없음")
        raise HTTPException(status_code=404, detail=f"'{query.prompt_name}' 프롬프트를 찾을 수 없습니다.")
    print(f"  - 조회된 프롬프트 이름: '{query.prompt_name}'")

    try:
        print("✅ [3/5] RAG 파이프라인 호출 시작...")
        answer = rag_pipeline.ask_modular_rag(
            purpose=query.purpose,
            question=query.question,
            prompt_template=prompt_template
        )
        print("✅ [4/5] 답변 생성 성공!")

        # 응답 후 백그라운드에서 메모리 정리
        background_tasks.add_task(cleanup_gpu)
        print("✅ [5/5] 클라이언트에 응답 전송 및 백그라운드 작업 예약 완료.")
        return QueryResponse(answer=answer)

    except Exception as e:
        # ===================================================================
        # 결정적인 단서! 에러가 발생하면 상세 내용을 모두 출력
        # ===================================================================
        print("\n" + "!"*50)
        print("❌ CRITICAL: 서버 내부에서 심각한 오류 발생!")
        print("!"*50)
        traceback.print_exc() # 에러의 상세한 원인을 모두 출력
        print("!"*50 + "\n")

        background_tasks.add_task(cleanup_gpu) # 에러 발생 시에도 메모리 정리 시도
        raise HTTPException(status_code=500, detail=f"서버 내부 처리 중 심각한 오류 발생: {e}")

# ... (ask-integrated 엔드포인트는 일단 생략) ...

# --- 서버 실행 ---
if __name__ == "__main__":
    # ===================================================================
    # 변경점 2: ngrok 연결 시 domain 옵션 사용
    # ===================================================================
    if not NGROK_STATIC_DOMAIN or "YOUR_STATIC_DOMAIN_HERE" in NGROK_STATIC_DOMAIN:
        print("경고: ngrok 고정 주소(NGROK_STATIC_DOMAIN)가 설정되지 않았습니다. 임의의 주소로 실행됩니다.")
        public_url = ngrok.connect(8000)
    else:
        public_url = ngrok.connect(8000, domain=NGROK_STATIC_DOMAIN)
        print(f"서버가 다음 고정 주소에서 실행 중입니다: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
