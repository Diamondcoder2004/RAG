# answer_generator.py
import requests
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ContextChunk:
    text: str
    source: str | None = None


class ChatTemplate:
    SYSTEM_PROMPT = """\
Ты — юридический помощник.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет ответа — прямо скажи "В контексте нет ответа".
Не выдумывай.

Контекст:
{context}
"""
    USER_PROMPT = "{question}"

    def render(self, question: str, contexts: List[ContextChunk]) -> List[dict]:
        context_text = "\n\n".join(
            f"[Источник {i + 1}]\n{c.text}"
            for i, c in enumerate(contexts)
        )

        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.format(context=context_text),
            },
            {
                "role": "user",
                "content": self.USER_PROMPT.format(question=question),
            },
        ]


class OllamaChatModel:
    def __init__(
            self,
            model: str = "qwen2.5:3b",
            host: str = "http://localhost:11434",
            temperature: float = 0.2,
            max_tokens: int = 512,
    ):
        self.model = model
        self.url = f"{host}/api/chat"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        resp = requests.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        return data["message"]["content"]


class RAGAnswerGenerator:
    def __init__(
            self,
            retriever,  # Любой ретривер с методом retrieve()
            chat_model: OllamaChatModel,
            template: ChatTemplate,
            max_contexts: int = 5,
    ):
        self.retriever = retriever
        self.chat_model = chat_model
        self.template = template
        self.max_contexts = max_contexts

    def answer(self, question: str) -> str:
        retrieved = self.retriever.retrieve(question)

        contexts = []
        for i, r in enumerate(retrieved[:self.max_contexts]):
            # Получаем source из payload
            source = None
            if isinstance(r.payload, dict):
                source = r.payload.get("source") or r.payload.get("url")

            contexts.append(ContextChunk(
                text=r.text,
                source=source
            ))

        messages = self.template.render(question, contexts)
        return self.chat_model.chat(messages)