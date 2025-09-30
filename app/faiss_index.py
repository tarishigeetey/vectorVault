import numpy as np
import faiss
from app.db_supabase import VectorDB

class FAISSIndex:
    """
    In-memory FAISS index for ANN search with optional metadata filtering
    """
    def __init__(self, vector_db: VectorDB):
        self.db = vector_db
        self.index = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.vectors = []
        self.metadata = []  # Store metadata for each vector
        self.build_index()

    def build_index(self):
        all_vectors = self.db._get_all()
        if not all_vectors:
            self.index = None
            self.metadata = []
            return

        dim = len(all_vectors[0]["embedding"])
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = np.array([v["embedding"] for v in all_vectors], dtype=np.float32)
        self.index.add(self.vectors)

        self.id_to_idx = {v["id"]: i for i, v in enumerate(all_vectors)}
        self.idx_to_id = {i: v["id"] for i, v in enumerate(all_vectors)}
        self.metadata = [v["meta"] for v in all_vectors]

    def search(self, query_vec, top_k=3, meta_filter=None):
        """
        Search for top_k nearest neighbors with optional metadata filter
        """
        if self.index is None:
            return []

        query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k * 5)  # extra to filter by metadata
        results = []

        for idx in I[0]:
            if idx == -1:
                continue
            vector_meta = self.metadata[idx]
            # Apply metadata filter
            if meta_filter:
                if not all(k in vector_meta and vector_meta[k] == v for k, v in meta_filter.items()):
                    continue
            vec_id = self.idx_to_id[idx]
            vec_data = next(v for v in self.db._get_all() if v["id"] == vec_id)
            results.append({"score": float(D[0][np.where(I[0]==idx)[0][0]]), **vec_data})
            if len(results) >= top_k:
                break

        return results
