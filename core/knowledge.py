"""Embeddings, clustering, 2-D projection, chunk index, retrieval."""
from __future__ import annotations

import functools
import re
from collections import Counter

import numpy as np

from core.models import Paper


@functools.lru_cache(maxsize=1)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


class Embedder:
    """Sentence-transformers, falling back to TF-IDF if the model can't load
    (e.g. OOM on Streamlit Cloud). `active` tells the UI which one is live."""

    def __init__(self):
        try:
            self._model = _load_sentence_transformer()
            self.active = "all-MiniLM-L6-v2"
        except Exception:
            self._model = None
            self.active = "tfidf-fallback"
        self._vectorizer = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._vectorizer = TfidfVectorizer(max_features=4096)
        vecs = self._vectorizer.fit_transform(texts).toarray()
        return normalize(vecs)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        from sklearn.preprocessing import normalize

        vecs = self._vectorizer.transform(texts).toarray()
        return normalize(vecs)


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def _sentence_chunks(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > size:
            chunks.append(current.strip())
            current = (current[-overlap:] + " " + sent).strip()
        else:
            current = (current + " " + sent).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _renumber_by_size(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist())
    order = [lbl for lbl, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])


class KnowledgeBase:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.papers: list[Paper] = []
        self.chunks: list[dict] = []
        self.chunk_vectors: np.ndarray | None = None
        self.paper_vectors: np.ndarray | None = None

    def build(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.chunks = []
        for i, p in enumerate(papers):
            self.chunks.append(
                {"paper_idx": i, "text": f"{p.title}. {p.abstract}".strip(), "primary": True}
            )
            if p.full_text:
                for piece in _sentence_chunks(p.full_text):
                    self.chunks.append({"paper_idx": i, "text": piece, "primary": False})

        texts = [c["text"] for c in self.chunks]
        self.chunk_vectors = self.embedder.fit_transform(texts)
        primary_rows = [i for i, c in enumerate(self.chunks) if c["primary"]]
        self.paper_vectors = self.chunk_vectors[primary_rows]

    def cluster(self, k: int) -> np.ndarray:
        from sklearn.cluster import KMeans

        n = len(self.papers)
        k = max(1, min(k, n))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = _renumber_by_size(km.fit_predict(self.paper_vectors))
        for i, p in enumerate(self.papers):
            p.cluster = int(labels[i])
        return labels

    def project_2d(self) -> np.ndarray:
        from sklearn.decomposition import PCA

        n_components = max(1, min(2, self.paper_vectors.shape[0], self.paper_vectors.shape[1]))
        coords = PCA(n_components=n_components).fit_transform(self.paper_vectors)
        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
        return coords

    def centroid_nearest(self, cluster_id: int, n: int) -> list[Paper]:
        idxs = [i for i, p in enumerate(self.papers) if p.cluster == cluster_id]
        if not idxs:
            return []
        vecs = self.paper_vectors[idxs]
        centroid = vecs.mean(axis=0)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        ranked = [idxs[i] for i in np.argsort(dists)]
        return [self.papers[i] for i in ranked[:n]]

    def search(
        self, query: str, k: int = 5, theme: int | None = None, max_per_paper: int = 2
    ) -> list[dict]:
        if not self.chunks:
            return []
        q_vec = self.embedder.transform([query])[0]
        candidate_idxs = [
            i
            for i, c in enumerate(self.chunks)
            if theme is None or self.papers[c["paper_idx"]].cluster == theme
        ]
        if not candidate_idxs:
            return []
        sims = self.chunk_vectors[candidate_idxs] @ q_vec
        order = np.argsort(-sims)

        results = []
        per_paper_count: dict[int, int] = {}
        for rank in order:
            idx = candidate_idxs[rank]
            chunk = self.chunks[idx]
            pidx = chunk["paper_idx"]
            if per_paper_count.get(pidx, 0) >= max_per_paper:
                continue
            per_paper_count[pidx] = per_paper_count.get(pidx, 0) + 1
            results.append({"paper": self.papers[pidx], "text": chunk["text"], "score": float(sims[rank])})
            if len(results) >= k:
                break
        return results


def label_themes_tfidf(papers: list[Paper]) -> dict[int, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    clusters = sorted({p.cluster for p in papers if p.cluster is not None})
    labels: dict[int, str] = {}
    for c in clusters:
        texts = [f"{p.title} {p.abstract}" for p in papers if p.cluster == c]
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=2000)
            matrix = vec.fit_transform(texts)
            scores = matrix.sum(axis=0).A1
            terms = vec.get_feature_names_out()
            top = [terms[i] for i in scores.argsort()[::-1][:3]]
            labels[c] = ", ".join(t.title() for t in top) if top else f"Theme {c + 1}"
        except ValueError:
            labels[c] = f"Theme {c + 1}"
    return labels
