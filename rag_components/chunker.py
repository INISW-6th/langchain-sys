from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import copy
# 1. 텍스트 데이터(JSON 또는 Document 객체) 입력
# 2. chunk_size, chunk_overlap 기준으로 분할
# 3. 줄바꿈, 마침표, 공백 등으로 텍스트 나눔
# 4. 메타데이터 유지한 Document 객체로 저장
# 5. 분할된 Document 리스트 반환
class MetadataChunkGenerator:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", "。", " ", ""], length_function=len)

    def generate_chunks(self, json_data: List[Dict]) -> List[Document]:
        all_chunks = []
        for item in json_data:
            if isinstance(item, Document):
                content, metadata = item.page_content, copy.deepcopy(item.metadata)
            else:
                content, metadata = item["content"], copy.deepcopy(item["metadata"])

            base_doc = Document(page_content=content, metadata=metadata)
            if len(content) > self.chunk_size:
                all_chunks.extend(self.splitter.split_documents([base_doc]))
            else:
                all_chunks.append(base_doc)
        return all_chunks
