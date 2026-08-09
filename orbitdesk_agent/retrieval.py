from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from orbitdesk_agent.state import RetrievedDocument


DOCUMENT_ID_PATTERN = re.compile(r"document_id:\s*(KB-\d+)")
TITLE_PATTERN = re.compile(r"title:\s*(.+)")


@dataclass(frozen=True)
class Chunk:
    source_id: str
    title: str
    passage: str


class LocalRetriever:
    def __init__(self, data_dir: Path, model_name: str, revision: str, offline: bool) -> None:
        self.model = SentenceTransformer(model_name, revision=revision, local_files_only=offline)
        self.model_name = model_name
        self.resolved_revision = self.model[0].auto_model.config._commit_hash or revision
        self.chunks = self._load_chunks(data_dir)
        passages = [chunk.passage for chunk in self.chunks]
        self.embeddings = self.model.encode(passages, normalize_embeddings=True)

    def search(self, query: str, limit: int = 3) -> list[RetrievedDocument]:
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_embedding)
        best_indices = np.argsort(scores)[::-1][:limit]
        return [
            {
                "source_id": self.chunks[index].source_id,
                "title": self.chunks[index].title,
                "passage": self.chunks[index].passage,
                "score": round(float(scores[index]), 4),
            }
            for index in best_indices
        ]

    @staticmethod
    def _load_chunks(data_dir: Path) -> list[Chunk]:
        knowledge_base = data_dir / "knowledge_base"
        chunks: list[Chunk] = []
        for path in sorted(knowledge_base.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            document_id = DOCUMENT_ID_PATTERN.search(text)
            title = TITLE_PATTERN.search(text)
            source_id = document_id.group(1) if document_id else path.name
            document_title = title.group(1) if title else path.stem
            sections = [section.strip() for section in text.split("\n## ") if section.strip()]
            for section in sections:
                chunks.append(Chunk(source_id, document_title, section[:1800]))
        if not chunks:
            raise ValueError(f"No Markdown knowledge-base files found in {knowledge_base}")
        return chunks
