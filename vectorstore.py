import faiss
from sentence_transformers import SentenceTransformer

class vector_store :
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)
        self.index=None
        self.documents=[]

    def build_index(self,docs,metadata=None):
        embeddings=self.model.encode(docs , convert_to_numpy=True)
        dim=embeddings.shape[1]
        self.index=faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadata:
            self.meta = metadata
        else:
            self.meta = docs

    def search(self, query, k=5):
        """Search in FAISS index and return (meta, distance, id)."""
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append((self.meta[idx], float(distances[0][rank]), int(idx)))
        return results
    