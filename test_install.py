# test_install.py
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

print("NumPy version:", np.__version__)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("HuggingFace model loaded successfully!")
app = FastAPI()
print("FastAPI imported successfully!")
