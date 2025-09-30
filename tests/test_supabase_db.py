import pytest
from app.db_supabase import VectorDB

# ----------------------
# Fixture: fresh DB instance
# ----------------------
@pytest.fixture
def db():
    """
    Fresh DB instance for each test
    Clears both vectors and vectors_history
    """
    db_instance = VectorDB()
    
    # Clear history first
    db_instance.history.supabase.table("vectors_history").delete().neq("version_id", 0).execute()
    
    # Then clear vectors
    db_instance.history.supabase.table("vectors").delete().neq("id", "").execute()
    
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

    # Check search returns updated text
    results = db.search("New")
    assert any(r["id"] == entry_id and r["text"] == "New text" for r in results)

    # Check history exists
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
