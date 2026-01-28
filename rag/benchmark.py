# benchmark.py
import csv
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm

from qdrant_client import QdrantClient
from semantic_router.encoders import HuggingFaceEncoder

from retriever import DenseRetriever, Reranker, HybridRetriever
from rag.answer_generator import OllamaChatModel, ChatTemplate, RAGAnswerGenerator

FAQ_PATH = Path("../faq_question.csv")
OUTPUT_PATH = Path("../benchmark_results.json")

QDRANT_COLLECTIONS = ["docs_hybrid"]
TOP_K_DENSE = 20
TOP_K_RERANK = 5
MIN_RERANK_SCORE = 0.15
RECALL_THRESHOLD = 0.65
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TEMP = 0.20
OLLAMA_MAX_TOKENS = 768

EMBED_MODEL_PATH = r"D:\hf_models\sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
RERANK_MODEL_PATH = r"D:\hf_models\BAAI__bge-reranker-base"
ENCODER_RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Используем GPU

def cosine_sim(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_faq(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def create_retriever() -> HybridRetriever:
    qdrant = QdrantClient(host="localhost", port=6333)
    encoder = HuggingFaceEncoder(name=EMBED_MODEL_PATH, device=ENCODER_RERANKER_DEVICE)

    dense_retrievers = []
    for collection in QDRANT_COLLECTIONS:
        dense = DenseRetriever(
            qdrant=qdrant,
            encoder=encoder,
            collection=collection,
            top_k=TOP_K_DENSE,
        )
        dense_retrievers.append(dense)

    reranker = Reranker(
        model_name=RERANK_MODEL_PATH,
        top_n=TOP_K_RERANK,
        device=ENCODER_RERANKER_DEVICE,
    )

    return HybridRetriever(
        retrievers=dense_retrievers,
        reranker=reranker,
        min_score=MIN_RERANK_SCORE,
    )

def run_benchmark():
    faq = load_faq(FAQ_PATH)
    if not faq:
        print("Файл FAQ пустой или не найден")
        return

    retriever = create_retriever()
    chat_model = OllamaChatModel(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMP,
        max_tokens=OLLAMA_MAX_TOKENS,
    )
    template = ChatTemplate()

    rag_generator = RAGAnswerGenerator(
        retriever=retriever,
        chat_model=chat_model,
        template=template,
        max_contexts=TOP_K_RERANK,
    )

    results = []
    coverage_results = []  # Новый раздел для проверки покрытия

    encoder = HuggingFaceEncoder(name=EMBED_MODEL_PATH, device=ENCODER_RERANKER_DEVICE)  # Для similarity

    for row in tqdm(faq, desc="Оценка RAG"):
        question = row["Вопрос"].strip()
        gt_answer = row["Ответ"].strip()

        if not question or not gt_answer:
            continue

        # Получаем чанки
        retrieved_chunks = retriever.retrieve(question)

        if not retrieved_chunks:
            results.append({
                "question": question,
                "status": "no_context",
                "answer_similarity": 0.0,
                "recall": 0.0,
                "max_chunk_sim": 0.0,
                "avg_chunk_sim": 0.0,
                "context_chunks": 0,
            })
            coverage_results.append({"question": question, "covered": False, "max_score": 0.0})
            continue

        # Генерируем ответ (как раньше)
        generated_answer = rag_generator.answer(question)

        # Оценка качества ответа (как раньше)
        emb_gen = encoder([generated_answer])[0]
        emb_gt = encoder([gt_answer])[0]
        answer_sim = cosine_sim(emb_gen, emb_gt)

        # Оценка retrieval (как раньше)
        chunk_texts = [r.text for r in retrieved_chunks]
        chunk_embs = encoder(chunk_texts)
        chunk_sims = [cosine_sim(emb, emb_gt) for emb in chunk_embs]

        max_chunk_sim = max(chunk_sims) if chunk_sims else 0.0
        avg_chunk_sim = float(np.mean(chunk_sims)) if chunk_sims else 0.0
        recall = 1.0 if max_chunk_sim >= RECALL_THRESHOLD else 0.0

        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": gt_answer,
            "answer_similarity": round(answer_sim, 4),
            "recall": recall,
            "max_chunk_sim": round(max_chunk_sim, 4),
            "avg_chunk_sim": round(avg_chunk_sim, 4),
            "context_chunks": len(retrieved_chunks),
        })

        # Проверка покрытия: если max_chunk_sim > 0.7 — покрыто
        covered = max_chunk_sim > 0.7
        coverage_results.append({"question": question, "covered": covered, "max_score": round(max_chunk_sim, 4)})

    # Сохранение результатов (как раньше)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Новый: Сохранение покрытия в отдельный файл
    coverage_path = Path("../coverage_report.json")
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage_results, f, ensure_ascii=False, indent=2)

    # Сводка (расширенная)
    if results:
        valid = [r for r in results if "answer_similarity" in r]

        avg_answer_sim = np.mean([r["answer_similarity"] for r in valid]) if valid else 0
        avg_recall = np.mean([r["recall"] for r in valid]) if valid else 0
        avg_chunks = np.mean([r["context_chunks"] for r in valid]) if valid else 0
        coverage_rate = np.mean([1 if c["covered"] else 0 for c in coverage_results])  # % покрытых вопросов

        print("\n" + "═" * 60)
        print(f"  Завершён бенчмарк → {OUTPUT_PATH}")
        print(f"  Покрытие корпуса: {coverage_rate:.3f} (доля вопросов с релевантными чанками)")
        print(f"  Вопросов обработано: {len(results)}")
        print(f"  Среднее сходство ответов   : {avg_answer_sim:.3f}")
        print(f"  Средний recall (hit-rate)  : {avg_recall:.3f}")
        print(f"  Среднее кол-во чанков      : {avg_chunks:.1f}")
        print("═" * 60 + "\n")

if __name__ == "__main__":
    run_benchmark()