from pydantic import BaseModel
from typing import List, Optional, Dict

class AddRequest(BaseModel):
    text: str
    meta: Optional[Dict] = None

class BatchAddRequest(BaseModel):
    texts: List[str]
    metas: Optional[List[Dict]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    meta_filter: Optional[Dict] = None

class DeleteRequest(BaseModel):
    entry_id: str

class UpdateRequest(BaseModel):
    entry_id: str
    text: Optional[str] = None
    meta: Optional[Dict] = None
