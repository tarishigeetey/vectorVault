import numpy as np
import uuid
from sentence_transformers import SentenceTransformer
from app.supabase_client import supabase
from app.vector_history import VectorHistory

class VectorDB:
    """
    VectorVault Database with Supabase backend
    - Stores embeddings + metadata
    - Supports unique IDs
    - Supports CRUD and search with optional metadata filters
    - Tracks version history via VectorHistory
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.history = VectorHistory(supabase)  # inject versioning

    # ----------------------
    # Add / Batch Add
    # ----------------------
    def add(self, text, meta=None):
        entry_id = str(uuid.uuid4())
        vector = self.model.encode(text, normalize_embeddings=True)
        vector_list = vector.tolist()  # JSON-serializable

        if meta is None:
            meta = {}

        data = {
            "id": entry_id,
            "text": text,
            "meta": meta,
            "embedding": vector_list
        }
        supabase.table("vectors").insert(data).execute()
        return entry_id

    def batch_add(self, texts, metas=None):
        ids = []
        for i, text in enumerate(texts):
            meta = metas[i] if metas else None
            ids.append(self.add(text, meta))
        return ids

    # ----------------------
    # Fetch all vectors (with optional metadata filter)
    # ----------------------
    def _get_all(self, meta_filter=None):
        res = supabase.table("vectors").select("*").execute()
        all_vectors = []

        for row in res.data:
            if meta_filter:
                if not all(k in row["meta"] and row["meta"][k] == v for k, v in meta_filter.items()):
                    continue
            embedding = np.array(row["embedding"], dtype=np.float32)
            all_vectors.append({
                "id": row["id"],
                "text": row["text"],
                "meta": row["meta"],
                "embedding": embedding
            })
        return all_vectors

    # ----------------------
    # Search
    # ----------------------
    def search(self, query, top_k=3, meta_filter=None):
        q_vec = self.model.encode(query, normalize_embeddings=True)
        all_vectors = self._get_all(meta_filter=meta_filter)

        sims = []
        for v in all_vectors:
            sim = float(np.dot(q_vec, v["embedding"]) / (np.linalg.norm(q_vec) * np.linalg.norm(v["embedding"])))
            sims.append((sim, v))

        sims.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **v} for s, v in sims[:top_k]]

    # ----------------------
    # Update with versioning
    # ----------------------
    def update(self, entry_id, text=None, meta=None):
        # Fetch current vector
        res = supabase.table("vectors").select("*").eq("id", entry_id).execute()
        if not res.data:
            return False
        old = res.data[0]

        # Save old version via VectorHistory
        self.history.save_version(old)

        # Prepare new data
        data = {}
        if text:
            vector = self.model.encode(text, normalize_embeddings=True)
            data["embedding"] = vector.tolist()
            data["text"] = text
        if meta is not None:
            data["meta"] = meta

        if data:
            supabase.table("vectors").update(data).eq("id", entry_id).execute()
            return True
        return False

    # ----------------------
    # Delete
    # ----------------------
    def delete(self, entry_id):
        """
        Delete a vector and its history.
        Ensures no foreign key constraint violation occurs.
        """
        # Delete history first
        self.history.supabase.table("vectors_history").delete().eq("vector_id", entry_id).execute()

        # Then delete main vector
        res = supabase.table("vectors").delete().eq("id", entry_id).execute()
        return len(res.data) > 0
