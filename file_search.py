import sqlite3
from typing import List
from openai import OpenAI
from tqdm import tqdm
import PyPDF2
import os  
import pandas as pd
import numpy as np
import pickle

DB_PATH = "pdf_chunks.db"

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def search_pdf_chunks(query: str, top_k: int = 5):
    query_emb = get_embedding(query)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_name, page, chunk_id, text, embedding FROM pdf_chunks")
    results = []
    for row in c.fetchall():
        file_name, page, chunk_id, text, emb_blob = row
        emb = pickle.loads(emb_blob)
        sim = cosine_similarity(query_emb, emb)
        results.append((sim, file_name, page, chunk_id, text))
    conn.close()
    results.sort(reverse=True)
    return results[:top_k]

def process_pdf_to_db(file_path: str, chunk_size: int = 500):
    chunks = extract_pdf_chunks(file_path, chunk_size)
    for chunk in tqdm(chunks, desc=f"Processing {os.path.basename(file_path)}"):
        embedding = get_embedding(chunk["text"])
        save_chunk_to_db(chunk, embedding)

def process_all_pdfs_to_db():
    init_db()
    # Get all PDF files from the assets directory
    pdf_files = [os.path.join("assets", f) for f in os.listdir("assets") if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        process_pdf_to_db(pdf_file)

def get_embedding(text: str) -> list:
    # Get embedding from OpenAI API
    client = OpenAI()  # Initialize the OpenAI client
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def save_chunk_to_db(chunk: dict, embedding: list):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO pdf_chunks (file_name, page, chunk_id, text, embedding) VALUES (?, ?, ?, ?, ?)",
        (
            chunk["file_name"],
            chunk["page"],
            chunk["chunk_id"],
            chunk["text"],
            pickle.dumps(embedding)
        )
    )
    conn.commit()
    conn.close()
    
def extract_pdf_chunks(file_path: str, chunk_size: int = 500) -> List[dict]:
    reader = PyPDF2.PdfReader(file_path)
    file_name = os.path.basename(file_path)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Split into chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]
            chunks.append({
                "file_name": file_name,
                "page": page_num,
                "chunk_id": i // chunk_size,
                "text": chunk_text
            })
    return chunks

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            page INTEGER,
            chunk_id INTEGER,
            text TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF Search App (SQLite + OpenAI API)")
    parser.add_argument("--register", action="store_true", help="Register PDFs to database")
    parser.add_argument("--search", type=str, help="Search with query")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to show")
    args = parser.parse_args()

    if args.register:
        print("Registering PDFs to database...")
        process_all_pdfs_to_db()
        print("Registration complete")
    elif args.search:
        print(f"Query: {args.search}")
        results = search_pdf_chunks(args.search, top_k=args.topk)
        for i, (sim, file_name, page, chunk_id, text) in enumerate(results, 1):
            print(f"\n--- Top {i} ---")
            print(f"Similarity: {sim:.4f}")
            print(f"File: {file_name}, Page: {page}, Chunk: {chunk_id}")
            print(f"Content: {text[:200]}...")
    else:
        parser.print_help()