import pytest
import os
import json
import csv
from app.db_supabase import VectorDB
from app.faiss_index import FAISSIndex

# ----------------------
# Fixture: fresh DB instance
# ----------------------
@pytest.fixture
def db():
    """
    Fresh DB instance for each test.
    Clears vectors and vectors_history before each test.
    """
    db_instance = VectorDB(use_faiss=True)
    
    # Clear history first
    db_instance.history.supabase.table("vectors_history").delete().neq("version_id", 0).execute()
    
    # Then clear vectors
    db_instance.history.supabase.table("vectors").delete().neq("id", "").execute()
    
    # Rebuild FAISS index after clearing
    db_instance._build_faiss_index()
    
    return db_instance

# ----------------------
# Test single add
# ----------------------
def test_add(db):
    entry_id = db.add("Cats are cute", meta={"category": "animals"})
    results = db.search("feline")
    assert any(r["id"] == entry_id for r in results)
    assert any(r["meta"]["category"] == "animals" for r in results)

# ----------------------
# Test batch add
# ----------------------
def test_batch_add(db):
    texts = ["AI is the future", "Python is versatile"]
    metas = [{"category": "tech"}, {"category": "programming"}]
    ids = db.batch_add(texts, metas)
    
    results1 = db.search("AI")
    results2 = db.search("Python")
    
    assert any(r["id"] == ids[0] for r in results1)
    assert any(r["id"] == ids[1] for r in results2)

# ----------------------
# Test search with metadata filter
# ----------------------
def test_search_filter(db):
    db.add("Machine learning research", meta={"category": "tech"})
    results = db.search("ML", meta_filter={"category": "tech"})
    
    assert len(results) > 0
    assert all(r["meta"]["category"] == "tech" for r in results)

# ----------------------
# Test update (with versioning)
# ----------------------
def test_update(db):
    entry_id = db.add("Old text", meta={"category": "test"})
    updated = db.update(entry_id, text="New text", meta={"category": "updated"})
    assert updated

    results = db.search("New")
    assert any(r["id"] == entry_id and r["text"] == "New text" for r in results)

    history = db.history.get_history(entry_id)
    assert len(history) == 1
    assert history[0]["text"] == "Old text"

# ----------------------
# Test delete
# ----------------------
def test_delete(db):
    entry_id = db.add("To be deleted", meta={"category": "temp"})
    deleted = db.delete(entry_id)
    assert deleted

    results = db.search("deleted")
    assert not any(r["id"] == entry_id for r in results)

# ----------------------
# Test version history multiple updates
# ----------------------
def test_multiple_updates_history(db):
    entry_id = db.add("Initial", meta={"category":"vtest"})
    db.update(entry_id, text="Second update")
    db.update(entry_id, text="Third update", meta={"category":"final"})
    
    history = db.history.get_history(entry_id)
    assert len(history) == 2
    texts = [h["text"] for h in history]
    assert "Initial" in texts
    assert "Second update" in texts

# ----------------------
# Test CSV batch insert
# ----------------------
def test_batch_add_csv(tmp_path, db):
    csv_file = tmp_path / "test.csv"
    fieldnames = ["text", "category"]
    rows = [
        {"text": "CSV entry 1", "category": "csv"},
        {"text": "CSV entry 2", "category": "csv"}
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    ids = db.batch_add_from_csv(str(csv_file), text_col="text", meta_cols=["category"])
    results = db.search("CSV")
    
    assert all(any(r["id"] == i for r in results) for i in ids)

# ----------------------
# Test JSON batch insert
# ----------------------
def test_batch_add_json(tmp_path, db):
    json_file = tmp_path / "test.json"
    data = [
        {"text": "JSON entry 1", "category": "json"},
        {"text": "JSON entry 2", "category": "json"}
    ]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    ids = db.batch_add_from_json(str(json_file), text_key="text", meta_keys=["category"])
    results = db.search("JSON")
    
    assert all(any(r["id"] == i for r in results) for i in ids)

# ----------------------
# Fixture: fresh DB + FAISS
# ----------------------
@pytest.fixture
def db_and_faiss():
    db = VectorDB()
    
    # Clear history first
    db.history.supabase.table("vectors_history").delete().neq("version_id", 0).execute()
    
    # Clear main vectors
    db.history.supabase.table("vectors").delete().neq("id", "").execute()
    
    # Add some initial vectors
    texts = ["cat", "dog", "apple", "orange", "car", "bike"]
    ids = db.batch_add(texts)
    
    # Build FAISS index
    faiss_index = FAISSIndex(db)
    
    return db, faiss_index, ids

# ----------------------
# Test FAISS nearest neighbor search
# ----------------------
def test_faiss_search(db_and_faiss):
    db, faiss_index, ids = db_and_faiss
    
    query_vec = db.model.encode("feline", normalize_embeddings=True)
    results = faiss_index.search(query_vec, top_k=2)
    
    top_texts = [r["text"] for r in results]
    assert "cat" in top_texts

# ----------------------
# Test FAISS index updates after adding new vector
# ----------------------
def test_faiss_after_add(db_and_faiss):
    db, faiss_index, ids = db_and_faiss
    
    new_id = db.add("kitten")
    faiss_index.build_index()
    
    query_vec = db.model.encode("feline", normalize_embeddings=True)
    results = faiss_index.search(query_vec, top_k=2)
    
    top_texts = [r["text"] for r in results]
    assert any(t in ["cat", "kitten"] for t in top_texts)

# ----------------------
# Test FAISS index updates after delete
# ----------------------
def test_faiss_after_delete(db_and_faiss):
    db, faiss_index, ids = db_and_faiss
    
    db.delete(ids[0])
    faiss_index.build_index()
    
    query_vec = db.model.encode("feline", normalize_embeddings=True)
    results = faiss_index.search(query_vec, top_k=2)
    
    top_texts = [r["text"] for r in results]
    assert "cat" not in top_texts

# ----------------------
# Test FAISS index update after modify
# ----------------------
def test_faiss_after_update(db_and_faiss):
    db, faiss_index, ids = db_and_faiss
    
    db.update(ids[1], text="puppy")
    faiss_index.build_index()
    
    query_vec = db.model.encode("puppy", normalize_embeddings=True)
    results = faiss_index.search(query_vec, top_k=2)
    
    top_texts = [r["text"] for r in results]
    assert "puppy" in top_texts
