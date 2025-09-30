from fastapi import FastAPI
from app.db import VectorDB
from app.schemas import AddRequest, BatchAddRequest, SearchRequest, DeleteRequest, UpdateRequest

app = FastAPI(title="VectorVault API")
db = VectorDB()  # Initialize the vector database

# ----------------------
# Add single entry
# ----------------------
@app.post("/add")
def add_item(req: AddRequest):
    entry_id = db.add(req.text, req.meta)
    return {"message": "Text added", "id": entry_id, "total_items": len(db.metadata)}

# ----------------------
# Batch add entries
# ----------------------
@app.post("/batch_add")
def batch_add(req: BatchAddRequest):
    ids = db.batch_add(req.texts, req.metas)
    return {"message": f"{len(req.texts)} texts added", "ids": ids, "total_items": len(db.metadata)}

# ----------------------
# Search entries
# ----------------------
@app.post("/search")
def search(req: SearchRequest):
    results = db.search(req.query, req.top_k, req.meta_filter)
    return {"query": req.query, "results": results}

# ----------------------
# Delete entry
# ----------------------
@app.post("/delete")
def delete(req: DeleteRequest):
    success = db.delete(req.entry_id)
    return {"success": success, "total_items": len(db.metadata)}

# ----------------------
# Update entry
# ----------------------
@app.post("/update")
def update(req: UpdateRequest):
    success = db.update(req.entry_id, req.text, req.meta)
    return {"success": success}

# ----------------------
# Save / Load DB
# ----------------------
@app.post("/save")
def save_db(path: str = "vector_db.pkl"):
    db.save(path)
    return {"message": f"Database saved to {path}"}

@app.post("/load")
def load_db(path: str = "vector_db.pkl"):
    db.load(path)
    return {"message": f"Database loaded from {path}", "total_items": len(db.metadata)}
