from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbedCluster:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
        self.metadata = []

    def fit(self, docs: List[str], metadata: List[Dict]):
        self.embeddings = self.model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
        self.index.add(self.embeddings)
        self.metadata = metadata

    def search(self, query: str, k: int = 5) -> List[Dict]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        return [self.metadata[i] | {"score": float(D[0][j])} for j, i in enumerate(I[0])]

    def kmeans(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None:
            raise ValueError("Call fit() first.")
        d = self.embeddings.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=25, verbose=False, spherical=True)
        kmeans.train(self.embeddings)
        D, I = kmeans.index.search(self.embeddings, 1)
        return I.reshape(-1), kmeans.centroids
