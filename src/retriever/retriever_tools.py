"""
search_kb()      – достаёт из базы знаний (docx-chunks) k самых близких пассов.
similar_case()   – вытаскивает исторический ответ оператора с похожим вопросом.

• Каждый инструмент использует **свой** заранее-обученный emb-модуль
  и отдельный FAISS-index.
• Отдаёт список словарей пригодный для prompt'а GPT-4o.
"""


import json, pathlib, functools
from dataclasses import dataclass

import faiss, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer


ROOT = pathlib.Path(__file__).resolve().parents[2]

KB_META      = ROOT / "data/processed/knowledge_blocks.parquet"
KB_INDEX     = ROOT / "data/processed/faiss_index.index"
KB_CASE_MODEL     = "intfloat/multilingual-e5-large"

CASE_DATA    = ROOT / "data/processed/q_a_transcripts.parquet"
CASE_INDEX   = ROOT / "data/processed/faiss_e5_transcripts.index"


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


@dataclass
class VectorSearcher:
    model_name: str
    index_path: pathlib.Path
    meta_path: pathlib.Path | None
    prefix_query: str = "query: "
    prefix_pass:  str = "passage: "

    def __post_init__(self):
        self.device = best_device()
        self.model  = SentenceTransformer(self.model_name, device=self.device)
        self.index  = faiss.read_index(str(self.index_path))

        if self.meta_path.suffix == ".parquet":
            self.meta = pd.read_parquet(self.meta_path)
        else:
            self.meta = list(load_jsonl(self.meta_path))

    def _embed(self, texts: list[str], is_query: bool = True):
        prefix = self.prefix_query if is_query else self.prefix_pass
        return self.model.encode([prefix + t for t in texts],
                                 normalize_embeddings=True)

    def search(self, query: str, k: int = 3):
        vec = self._embed([query])
        sims, idxs = self.index.search(np.asarray(vec, dtype="float32"), k)
        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.meta.iloc[idx] if isinstance(self.meta, pd.DataFrame) else self.meta[idx]
            results.append({"payload": meta, "sim": float(sim)})
        return results


@functools.lru_cache(maxsize=1)
def kb_searcher() -> VectorSearcher:
    return VectorSearcher(KB_CASE_MODEL, KB_INDEX, KB_META)


@functools.lru_cache(maxsize=1)
def case_searcher() -> VectorSearcher:
    return VectorSearcher(KB_CASE_MODEL, CASE_INDEX, CASE_DATA)


def search_kb(query: str, k: int = 3) -> list[dict]:
    """
    Найти k (=3) релевантных фрагментов базы знаний.
    Возвращает [{text, source, sim}] отсортированный по sim DESC.
    """
    hits = kb_searcher().search(query, k)
    out = []
    for h in hits:
        out.append({
            "text":   h["payload"]["content"],
            "source": h["payload"]["source"],
            "sim":    round(h["sim"], 3)
        })
    return out


def similar_case(query: str, k: int = 1) -> list[dict]:
    """
    Вернуть k (=1) исторических ответов оператора.
    [{answer, source, sim}]
    """
    hits = case_searcher().search(query, k)
    out = []
    for h in hits:
        row = h["payload"]
        out.append({
            "answer": row["operator_text"],
            "source": f"case:{row.name}" if isinstance(h["payload"], pd.Series) else row.get("id"),
            "sim":    round(h["sim"], 3)
        })
    return out
