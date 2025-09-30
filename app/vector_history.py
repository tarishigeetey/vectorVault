class VectorHistory:
    """
    Handles versioning of vectors in Supabase.
    Tracks all updates for undo/audit purposes.
    """
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    def save_version(self, vector_record):
        """
        Save a snapshot of the vector before updating.
        `vector_record` should contain: id, text, meta, embedding
        """
        data = {
            "vector_id": vector_record["id"],
            "text": vector_record["text"],
            "meta": vector_record.get("meta") or {},
            "embedding": vector_record["embedding"]
        }
        self.supabase.table("vectors_history").insert(data).execute()

    def get_history(self, vector_id):
        """
        Fetch all past versions of a vector, most recent first.
        """
        res = self.supabase.table("vectors_history")\
            .select("*")\
            .eq("vector_id", vector_id)\
            .order("updated_at", desc=True)\
            .execute()
        return res.data
