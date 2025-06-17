from typing import Dict, List, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_components.chunker import MetadataChunkGenerator
from rag_components.embedder import get_embedding_model
from rag_components.reranker import BgeReranker, CohereReranker
from rag_components.llm import get_hf_llm
from config import LLM_MODEL_MAP, SYSTEM_PROMPT_MAP
from langchain_community.vectorstores import FAISS

class ModularRAG:
    def __init__(self, config: Dict[str, Any], docs: Dict[str, List[Document]]):
        self.config, self.docs = config, docs
        self.rag_instances = {purpose: self._build_single_rag(docs) for purpose, docs in self.docs.items()}

    def _build_single_rag(self, docs: List[Document]) -> Dict:
        chunk_config = self.config.get("chunking", {})
        if chunk_config.get("method") == "custom":
            chunks = MetadataChunkGenerator(chunk_size=chunk_config.get("chunk_size"), chunk_overlap=chunk_config.get("chunk_overlap")).generate_chunks(docs)
        else:
            chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_config.get("chunk_size"), chunk_overlap=chunk_config.get("chunk_overlap")).split_documents(docs)
        embedding = get_embedding_model(self.config)
        vectorstore = FAISS.from_documents(chunks, embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.config.get("initial_top_k", 20)})
        reranker_type = self.config.get("reranker")
        reranker = BgeReranker() if reranker_type == "bge" else (CohereReranker(api_key=self.config["cohere_api_key"]) if reranker_type == "cohere" else None)
        llm = get_hf_llm(LLM_MODEL_MAP[self.config["llm"]], SYSTEM_PROMPT_MAP.get(self.config["llm"]))
        return {"retriever": retriever, "reranker": reranker, "llm": llm}

    def ask_modular_rag(self, purpose: str, question: str, prompt_template: str) -> str:
        if purpose not in self.rag_instances:
            raise ValueError(f"'{purpose}'에 해당하는 RAG 인스턴스를 찾을 수 없습니다.")
        rag = self.rag_instances[purpose]
        docs = rag["retriever"].get_relevant_documents(question)
        if rag["reranker"]:
            docs = rag["reranker"].rerank(question, docs, self.config.get("rerank_top_k", 5))
        context = "\\n\\n".join(doc.page_content for doc in docs[:self.config.get("rerank_top_k", 5)])
        final_prompt = prompt_template.format(context=context, question=question)
        return rag["llm"](final_prompt)

    def ask_naive_rag(self, purposes: List[str], question: str, prompt_template: str) -> str:
        for purpose in purposes:
            if purpose not in self.rag_instances:
                raise ValueError(f"'{purpose}'에 해당하는 RAG 인스턴스를 찾을 수 없습니다.")
        all_docs = []
        for purpose in purposes:
            rag = self.rag_instances[purpose]
            docs = rag["retriever"].get_relevant_documents(question)
            if rag["reranker"]:
                docs = rag["reranker"].rerank(question, docs, self.config.get("rerank_top_k", 5))
            all_docs.extend(docs[:self.config.get("rerank_top_k", 5)])
        context = "\\n\\n".join(doc.page_content for doc in list({doc.page_content: doc for doc in all_docs}.values())[:self.config.get("max_total_docs", 10)])
        final_prompt = prompt_template.format(context=context, question=question, purposes=", ".join(purposes))
        return self.rag_instances[purposes[0]]["llm"](final_prompt)
