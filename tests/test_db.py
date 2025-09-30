import pytest
from app.db import VectorDB

# ----------------------
# Fixture for a fresh DB
# ----------------------
@pytest.fixture
def db():
    return VectorDB()

# ----------------------
# Test Adding Single Entry
# ----------------------
def test_add(db):
    entry_id = db.add("Test text", meta={"category": "test"})
    assert len(db.metadata) == 1
    assert db.metadata[0]["id"] == entry_id
    assert db.metadata[0]["text"] == "Test text"
    assert db.metadata[0]["meta"]["category"] == "test"

# ----------------------
# Test Batch Add
# ----------------------
def test_batch_add(db):
    texts = ["Text 1", "Text 2"]
    metas = [{"category": "a"}, {"category": "b"}]
    ids = db.batch_add(texts, metas)
    assert len(db.metadata) == 2
    assert ids[0] == db.metadata[0]["id"]
    assert ids[1] == db.metadata[1]["id"]

# ----------------------
# Test Search
# ----------------------
def test_search(db):
    db.add("Cats are cute", meta={"category": "animals"})
    db.add("Dogs are loyal", meta={"category": "animals"})
    results = db.search("puppy", top_k=1)
    assert len(results) == 1
    assert "text" in results[0]
    assert "score" in results[0]

# ----------------------
# Test Search with Metadata Filter
# ----------------------
def test_search_filter(db):
    db.add("Tech article", meta={"category": "tech"})
    db.add("AI research", meta={"category": "AI"})
    results = db.search("AI", top_k=2, meta_filter={"category": "AI"})
    assert len(results) == 1
    assert results[0]["meta"]["category"] == "AI"

# ----------------------
# Test Update
# ----------------------
def test_update(db):
    entry_id = db.add("Old text", meta={"category": "test"})
    updated = db.update(entry_id, text="New text", meta={"category": "updated"})
    assert updated
    assert db.metadata[0]["text"] == "New text"
    assert db.metadata[0]["meta"]["category"] == "updated"

# ----------------------
# Test Delete
# ----------------------
def test_delete(db):
    entry_id = db.add("To be deleted")
    deleted = db.delete(entry_id)
    assert deleted
    assert len(db.metadata) == 0

# ----------------------
# Test Save & Load
# ----------------------
def test_save_load(tmp_path):
    db1 = VectorDB()
    db1.add("Persistent text", meta={"category": "persist"})
    file_path = tmp_path / "db.pkl"
    db1.save(file_path)

    db2 = VectorDB()
    db2.load(file_path)
    assert len(db2.metadata) == 1
    assert db2.metadata[0]["text"] == "Persistent text"
