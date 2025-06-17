import os
from openai import OpenAI
from typing import Dict, Tuple

# --- 평가 설정 (상수) ---

# 평가 지침
LLM_JUDGE_EVAL_GUIDE = """
* 평가 항목 및 배점
1. 역사 사실 정확도 (2점)
- 문서에서 확인 가능한 사실만 제시했는가?
- 역사적 사건, 연도, 인물 서술이 문서와 일치하는가?
2. 문서 참조 충실도 (2점)
- 교과서, 지도안, 민족대백과 등 다양한 문서를 균형 있게 인용했는가?
3. 구조적 완성도 (1점)
- 기(起)-승(承)-전(轉)-결 구조가 명확히 드러나는가?
4. 교육적 유용성 (1점)
- 중학생이 이해할 수 있도록 쉬운 문장과 어휘를 사용했는가?
5. 신화성 배제 (1점)
- 과장된 표현, 신비주의, 비사실적 상상 없이 작성되었는가?
** 평가 시 유의사항
... (이전 내용과 동일) ...
"""

# 조선의 건국에 대한 문서 자료
CONTEXT_TEXT = """
조선의 건국
⑴ 위화도 회군(1388): 명의 철령 이북 땅 요구 → 요동 정벌 추진 → 이성계가 위화도에서 회군 → 이성계의 국정 장악
... (이전 내용과 동일) ...
"""

# 정답 데이터 (GPT4 Judge 평가를 위한 reference)
REFERENCE_TEXT = """
기(起) - 발단
명의 철령위 설치 통보로 군사적 긴장이 고조되자, 고려는 요동 정벌을 결정하였다(출처:민족대백과_4)
... (이전 내용과 동일) ...
"""

# --- 평가 실행 함수 ---

def evaluate_answer_with_custom_judge(answer_to_grade: str) -> Tuple[float, str]:
    """
    하나의 답변을 받아 커스텀 LLM 평가자(GPT-4o)에게 채점을 요청하는 함수.
    
    Args:
        answer_to_grade: RAG 시스템이 생성한 평가 대상 답변.
        
    Returns:
        (점수, 채점 이유) 튜플. 오류 발생 시 (None, "오류 메시지").
    """
    # 함수가 호출될 때마다 클라이언트 초기화
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
너는 중학생을 위한 역사 교과서 감수 위원회 위원이다.
아래는 조선의 건국과정에 대한 LLM 답변이며, 제공된 문서(context 및 reference)에 기반해 평가해라.
평가지침에 따라 총 7점 만점으로 정량 평가하고, 각 항목별 감점 사유를 포함한 평가 이유를 설명하라.

[평가지침]
{LLM_JUDGE_EVAL_GUIDE}

[질문]
조선의 건국과정

[문서 context]
{CONTEXT_TEXT}

[문서 reference 요약]
{REFERENCE_TEXT}

[평가할 LLM 응답]
{answer_to_grade}

[출력 형식 예시]
점수: 6.5
이유:
- 역사적 사실은 전반적으로 문서와 일치하나 과전법 서술 누락 (-0.5)
- 교과서 표현과 유사하고 중학생 수준 어휘 사용
- 신화적 표현 없음, 구조적 흐름도 양호함
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 문서 기반으로 역사적 응답을 평가하는 전문가다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        # 점수와 이유 파싱
        score_line = next((line for line in content.splitlines() if line.strip().startswith("점수")), "")
        score = float(score_line.split(":")[1].strip()) if score_line else None
        
        reason_start_index = content.find("이유:")
        reason = content[reason_start_index:].strip() if reason_start_index != -1 else content

        return score, reason

    except Exception as e:
        print(f"커스텀 평가자 호출 중 오류 발생: {e}")
        return None, str(e)
