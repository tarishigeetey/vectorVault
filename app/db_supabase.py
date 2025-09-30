import numpy as np
import uuid
import csv
import json
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
    - Supports batch inserts and CSV/JSON uploads
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.history = VectorHistory(supabase)  # inject versioning
        self.supabase = supabase

    # ----------------------
    # Add / Batch Add
    # ----------------------
    def add(self, text, meta=None):
        """Add a single text vector"""
        entry_id = str(uuid.uuid4())
        vector = self.model.encode(text, normalize_embeddings=True).tolist()

        if meta is None:
            meta = {}

        data = {
            "id": entry_id,
            "text": text,
            "meta": meta,
            "embedding": vector
        }
        self.supabase.table("vectors").insert(data).execute()
        return entry_id

    def batch_add(self, texts, metas=None):
        """Add multiple texts in a single Supabase call"""
        data_batch = []
        for i, text in enumerate(texts):
            meta = metas[i] if metas else {}
            vector = self.model.encode(text, normalize_embeddings=True).tolist()
            entry_id = str(uuid.uuid4())
            data_batch.append({
                "id": entry_id,
                "text": text,
                "meta": meta,
                "embedding": vector
            })
        
        self.supabase.table("vectors").insert(data_batch).execute()
        return [d["id"] for d in data_batch]

    # ----------------------
    # CSV / JSON Uploads
    # ----------------------
    def batch_add_from_csv(self, csv_path, text_col="text", meta_cols=None):
        """Read CSV and insert all rows as vectors"""
        texts = []
        metas = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[text_col])
                meta = {col: row[col] for col in meta_cols} if meta_cols else {}
                metas.append(meta)
        
        return self.batch_add(texts, metas)

    def batch_add_from_json(self, json_path, text_key="text", meta_keys=None):
        """Read JSON file and insert all entries as vectors"""
        texts = []
        metas = []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                texts.append(item[text_key])
                meta = {k: item[k] for k in meta_keys} if meta_keys else {}
                metas.append(meta)

        return self.batch_add(texts, metas)

    # ----------------------
    # Fetch all vectors (with optional metadata filter)
    # ----------------------
    def _get_all(self, meta_filter=None):
        res = self.supabase.table("vectors").select("*").execute()
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
        res = self.supabase.table("vectors").select("*").eq("id", entry_id).execute()
        if not res.data:
            return False
        old = res.data[0]

        # Save old version via VectorHistory
        self.history.save_version(old)

        # Prepare new data
        data = {}
        if text:
            vector = self.model.encode(text, normalize_embeddings=True).tolist()
            data["embedding"] = vector
            data["text"] = text
        if meta is not None:
            data["meta"] = meta

        if data:
            self.supabase.table("vectors").update(data).eq("id", entry_id).execute()
            return True
        return False

    # ----------------------
    # Delete
    # ----------------------
    def delete(self, entry_id):
        """Delete a vector and its history"""
        # Delete history first
        self.history.supabase.table("vectors_history").delete().eq("vector_id", entry_id).execute()

        # Then delete main vector
        res = self.supabase.table("vectors").delete().eq("id", entry_id).execute()
        return len(res.data) > 0
