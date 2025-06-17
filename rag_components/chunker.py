from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import copy

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
