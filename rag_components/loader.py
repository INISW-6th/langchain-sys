import os, glob, json, re
from typing import Dict, List
from langchain.schema import Document
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
