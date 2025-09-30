import numpy as np
import pickle
import uuid
from sentence_transformers import SentenceTransformer

class VectorDB:
    """
    VectorVault Core Database
    - Stores embeddings + metadata
    - Supports unique IDs
    - Supports CRUD and search with optional metadata filters
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vectors = []   # List of numpy arrays
        self.metadata = []  # List of dicts: {"id": uuid, "text": str, "meta": dict}

    # ----------------------
    # Add / Batch Add
    # ----------------------
    def add(self, text, meta=None):
        """Add a single text entry and return its unique ID"""
        vector = self.model.encode(text, normalize_embeddings=True)
        entry_id = str(uuid.uuid4())
        self.vectors.append(vector)
        self.metadata.append({"id": entry_id, "text": text, "meta": meta})
        return entry_id

    def batch_add(self, texts, metas=None):
        """Add multiple texts at once"""
        ids = []
        for i, text in enumerate(texts):
            meta = metas[i] if metas else None
            ids.append(self.add(text, meta))
        return ids

    # ----------------------
    # Search
    # ----------------------
    def cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors"""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def search(self, query, top_k=3, meta_filter=None):
        """Search top_k results for a query with optional metadata filtering"""
        q_vec = self.model.encode(query, normalize_embeddings=True)
        sims = []
        filtered_indices = []

        # Apply metadata filter if provided
        for i, data in enumerate(self.metadata):
            if meta_filter:
                if not all(k in data["meta"] and data["meta"][k] == v for k, v in meta_filter.items()):
                    continue
            filtered_indices.append(i)
            sims.append(self.cosine_similarity(q_vec, self.vectors[i]))

        if not sims:
            return []

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.metadata[filtered_indices[i]] | {"score": sims[i]} for i in top_idx]

    # ----------------------
    # Delete / Update by ID
    # ----------------------
    def delete(self, entry_id):
        for i, data in enumerate(self.metadata):
            if data["id"] == entry_id:
                self.vectors.pop(i)
                self.metadata.pop(i)
                return True
        return False

    def update(self, entry_id, text=None, meta=None):
        for i, data in enumerate(self.metadata):
            if data["id"] == entry_id:
                if text:
                    self.vectors[i] = self.model.encode(text, normalize_embeddings=True)
                    self.metadata[i]["text"] = text
                if meta:
                    self.metadata[i]["meta"] = meta
                return True
        return False

    # ----------------------
    # Save / Load
    # ----------------------
    def save(self, path="vector_db.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "metadata": self.metadata}, f)

    def load(self, path="vector_db.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]
