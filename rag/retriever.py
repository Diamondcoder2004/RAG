# retriever.py
import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from sentence_transformers import CrossEncoder
from semantic_router.encoders import HuggingFaceEncoder

DEVICE = 'cpu'

@dataclass
class RetrievalResult:
    text: str
    score: float
    payload: Dict[str, Any]


class DenseRetriever:
    def __init__(
            self,
            qdrant: QdrantClient,
            encoder: HuggingFaceEncoder,
            collection: str,
            top_k: int = 20,
            payload_filter: Filter = None,
    ):
        self.qdrant = qdrant
        self.encoder = encoder
        self.collection = collection
        self.top_k = top_k
        self.payload_filter = payload_filter

    def retrieve(self, query: str) -> List[RetrievalResult]:
        query_vector = self.encoder([query])[0]

        search_result = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=self.top_k,
            query_filter=self.payload_filter,
            with_payload=True,
            with_vectors=False,
        ).points

        results = []
        for point in search_result:
            results.append(RetrievalResult(
                text=point.payload.get('text', ''),
                score=point.score,
                payload=point.payload
            ))

        return results


class Reranker:
    def __init__(
            self,
            model_name: str = r"D:\hf_models\BAAI__bge-reranker-base",
            top_n: int = 5,
            device: str = DEVICE,
    ):
        self.model = CrossEncoder(model_name, device=device)
        self.top_n = top_n

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return []

        pairs = [[query, r.text] for r in results]
        scores = self.model.predict(pairs)

        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for res, score in scored_results[:self.top_n]:
            # Создаем копию с обновленным скором
            reranked.append(RetrievalResult(
                text=res.text,
                score=float(score),
                payload=res.payload
            ))

        return reranked


class HybridRetriever:
    def __init__(
            self,
            retrievers: List[DenseRetriever],  # Изменено: принимает список ретриверов
            reranker: Reranker,
            min_score: float = 0.2,
    ):
        self.retrievers = retrievers
        self.reranker = reranker
        self.min_score = min_score

    def retrieve(self, query: str) -> List[RetrievalResult]:
        all_results = []
        for retriever in self.retrievers:
            all_results.extend(retriever.retrieve(query))

        if not all_results:
            return []

        # Удаление дубликатов по тексту
        unique_results = []
        seen_texts = set()
        for res in all_results:
            if res.text not in seen_texts:
                seen_texts.add(res.text)
                unique_results.append(res)

        # Реранкинг
        reranked = self.reranker.rerank(query, unique_results)

        # Фильтрация по минимальному скору
        filtered = [r for r in reranked if r.score >= self.min_score]

        return filtered