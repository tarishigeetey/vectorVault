import numpy as np
import uuid
import csv
import json
import faiss
from sentence_transformers import SentenceTransformer
from app.supabase_client import supabase
from app.vector_history import VectorHistory


class VectorDB:
    """
    VectorVault Database with Supabase backend
    Features:
    - Stores embeddings + metadata
    - Supports unique IDs
    - CRUD operations
    - Batch inserts & CSV/JSON uploads
    - Tracks version history via VectorHistory
    - Optional FAISS ANN search for fast queries
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", use_faiss=True):
        self.model = SentenceTransformer(model_name)
        self.history = VectorHistory(supabase)
        self.supabase = supabase
        self.use_faiss = use_faiss
        self.index = None
        self.id_map = []

        if use_faiss:
            self._build_faiss_index()

    # ----------------------
    # FAISS ANN index
    # ----------------------
    def _build_faiss_index(self):
        all_vectors = self._get_all()
        if not all_vectors:
            self.index = None
            self.id_map = []
            return

        dim = len(all_vectors[0]["embedding"])
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity via normalized vectors
        embeddings = np.array([v["embedding"] for v in all_vectors], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.id_map = [v["id"] for v in all_vectors]

    def _faiss_search(self, query_vec, top_k=3):
        if self.index is None or len(self.id_map) == 0:
            return []

        query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx >= len(self.id_map):
                continue
            vector_id = self.id_map[idx]
            row = self.supabase.table("vectors").select("*").eq("id", vector_id).execute().data[0]
            results.append({"score": float(score), **row})
        return results

    # ----------------------
    # Add / Batch Add
    # ----------------------
    def add(self, text, meta=None):
        entry_id = str(uuid.uuid4())
        vector = self.model.encode(text, normalize_embeddings=True).tolist()
        meta = meta or {}

        self.supabase.table("vectors").insert({
            "id": entry_id,
            "text": text,
            "meta": meta,
            "embedding": vector
        }).execute()

        if self.use_faiss:
            if self.index is None:
                self._build_faiss_index()
            else:
                vec_np = np.array([vector], dtype=np.float32)
                faiss.normalize_L2(vec_np)
                self.index.add(vec_np)
                self.id_map.append(entry_id)

        return entry_id

    def batch_add(self, texts, metas=None):
        metas = metas or [{} for _ in texts]
        data_batch = []
        vectors_batch = []

        for text, meta in zip(texts, metas):
            entry_id = str(uuid.uuid4())
            vector = self.model.encode(text, normalize_embeddings=True).tolist()
            data_batch.append({
                "id": entry_id,
                "text": text,
                "meta": meta,
                "embedding": vector
            })
            vectors_batch.append((entry_id, np.array(vector, dtype=np.float32)))

        if data_batch:
            self.supabase.table("vectors").insert(data_batch).execute()

        if self.use_faiss:
            if self.index is None and vectors_batch:
                self._build_faiss_index()
            else:
                vecs = np.array([v[1] for v in vectors_batch], dtype=np.float32)
                faiss.normalize_L2(vecs)
                self.index.add(vecs)
                self.id_map.extend([v[0] for v in vectors_batch])

        return [d["id"] for d in data_batch]

    # ----------------------
    # CSV / JSON Uploads
    # ----------------------
    def batch_add_from_csv(self, csv_path, text_col="text", meta_cols=None):
        texts, metas = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[text_col])
                metas.append({col: row[col] for col in meta_cols} if meta_cols else {})
        return self.batch_add(texts, metas)

    def batch_add_from_json(self, json_path, text_key="text", meta_keys=None):
        texts, metas = [], []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                texts.append(item[text_key])
                metas.append({k: item[k] for k in meta_keys} if meta_keys else {})
        return self.batch_add(texts, metas)

    # ----------------------
    # Fetch all vectors (with optional metadata filter)
    # ----------------------
    def _get_all(self, meta_filter=None):
        res = self.supabase.table("vectors").select("*").execute()
        all_vectors = []
        for row in res.data:
            if meta_filter and not all(k in row["meta"] and row["meta"][k] == v for k, v in meta_filter.items()):
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
        if self.use_faiss:
            results = self._faiss_search(q_vec, top_k)
            # Apply additional metadata filter if needed
            if meta_filter:
                results = [r for r in results if all(k in r["meta"] and r["meta"][k] == v for k, v in meta_filter.items())]
            return results
        else:
            all_vectors = self._get_all(meta_filter)
            sims = [(float(np.dot(q_vec, v["embedding"]) / 
                    (np.linalg.norm(q_vec) * np.linalg.norm(v["embedding"]))), v) for v in all_vectors]
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
        self.history.save_version(old)

        data = {}
        if text:
            data["text"] = text
            vector = self.model.encode(text, normalize_embeddings=True).tolist()
            data["embedding"] = vector
        if meta is not None:
            data["meta"] = meta

        if data:
            self.supabase.table("vectors").update(data).eq("id", entry_id).execute()
            # Update FAISS index
            if self.use_faiss and self.index and entry_id in self.id_map:
                idx = self.id_map.index(entry_id)
                vec_np = np.array([data["embedding"]], dtype=np.float32)
                faiss.normalize_L2(vec_np)
                self.index.reconstruct(idx)  # Optional: replace vector at idx
            return True
        return False

    # ----------------------
    # Delete
    # ----------------------
    def delete(self, entry_id):
        self.history.supabase.table("vectors_history").delete().eq("vector_id", entry_id).execute()
        res = self.supabase.table("vectors").delete().eq("id", entry_id).execute()
        if self.use_faiss and entry_id in self.id_map:
            idx = self.id_map.index(entry_id)
            self.index.remove_ids(np.array([idx]))
            self.id_map.pop(idx)
        return len(res.data) > 0
