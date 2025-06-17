import os, glob, json, re
from typing import Dict, List
from langchain.schema import Document
# 1. 지정된 경로에서 모든 JSON 파일 탐색  
# 2. 파일 이름에서 목적(purpose) 추출 (언더스코어 앞 부분)  
# 3. JSON 파일을 로드하여 텍스트와 메타데이터 가져옴  
# 4. 각 항목을 LangChain Document 객체로 변환하고 source_file 추가  
# 5. 목적별로 문서들을 딕셔너리 형태로 저장 및 반환  

def load_purpose_docs(data_path: str) -> Dict[str, List[Document]]:
    json_files = glob.glob(f"{data_path}/*.json")
    purpose_docs = {}
    for file_path in json_files:
        filename = os.path.basename(file_path)
        m = re.match(r"^([^_]+)_", filename)
        purpose = m.group(1) if m else os.path.splitext(filename)[0]
        with open(file_path, "r", encoding="utf-8-sig") as f:
            raw_data = json.load(f)
        if purpose not in purpose_docs:
            purpose_docs[purpose] = []
        purpose_docs[purpose].extend([Document(page_content=item["content"], metadata={**item["metadata"], "source_file": filename}) for item in raw_data])
    return purpose_docs
