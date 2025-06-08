import os
import sqlite3
import openai
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database setup
DB_PATH = "ukiyo_e.db"

def create_database():
    """Create SQLite database with ukiyo-e titles and their embeddings."""
    # Remove existing database file if exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for ukiyo-e titles and embeddings
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ukiyo_e (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT NOT NULL,
        period TEXT NOT NULL,
        description TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    ''')
    
    # Sample data: 10 ukiyo-e titles with artists, periods and descriptions
    ukiyo_e_data = [
        ("富嶽三十六景 神奈川沖浪裏", "葛飾北斎", "江戸時代後期", "大波と富士山の風景。北斎の代表作として世界的に有名。"),
        ("東都浅草本願寺", "安藤広重", "江戸時代後期", "江戸の浅草にある本願寺を描いた作品。"),
        ("歌麿の美人画 高島おひさ", "喜多川歌麿", "江戸時代中期", "美人画の名手・歌麿によって描かれた当時の美人像。"),
        ("東海道五十三次 日本橋", "安藤広重", "江戸時代後期", "東海道の起点である日本橋を描いた名作。"),
        ("冨嶽三十六景 凱風快晴", "葛飾北斎", "江戸時代後期", "赤富士とも呼ばれる、朝日に染まる富士山を描いた作品。"),
        ("南総里見八犬伝", "歌川国芳", "江戸時代後期", "人気小説を題材にした勇壮な武者絵。"),
        ("東都名所 浅草金龍山", "歌川豊国", "江戸時代後期", "浅草寺と雷門を描いた江戸の名所図。"),
        ("月に鷹", "葛飾応為", "江戸時代後期", "月光の下、枝に留まる鷹を描いた繊細な作品。"),
        ("見返り美人図", "菱川師宣", "江戸時代前期", "振り返る女性の姿を捉えた初期浮世絵の代表作。"),
        ("江戸花見の宴", "鳥居清長", "江戸時代中期", "桜の季節に宴を楽しむ江戸の人々を描いた作品。")
    ]
    
    # Clear existing data
    cursor.execute("DELETE FROM ukiyo_e")
    
    # Insert data and generate embeddings
    for title, artist, period, description in ukiyo_e_data:
        # Get embedding for the title and description
        full_text = f"Title: {title}, Artist: {artist}, Period: {period}, Description: {description}"
        embedding = get_embedding(full_text)
        
        # Convert embedding to binary for storage
        # OpenAI embeddings are float32 with 1536 dimensions
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        # Insert into database
        cursor.execute(
            "INSERT INTO ukiyo_e (title, artist, period, description, embedding) VALUES (?, ?, ?, ?, ?)",
            (title, artist, period, description, embedding_bytes)
        )
    
    conn.commit()
    conn.close()
    print(f"Database created with {len(ukiyo_e_data)} ukiyo-e entries")

def get_embedding(text):
    """Get embedding for a text using OpenAI's API."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def search_similar_ukiyo_e(query, top_k=3):
    """Search for ukiyo-e titles similar to the query."""
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all ukiyo-e data and embeddings
    cursor.execute("SELECT id, title, artist, period, description, embedding FROM ukiyo_e")
    results = cursor.fetchall()
    
    # Calculate similarity scores
    similarities = []
    for id, title, artist, period, description, embedding_bytes in results:
        # Convert bytes back to numpy array
        # OpenAI embeddings are float32 with 1536 dimensions
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(-1)
        
        # Make sure the dimensions match
        if len(embedding) != len(query_embedding):
            print(f"Warning: Embedding dimension mismatch: {len(embedding)} vs {len(query_embedding)}")
            continue
            
        # Calculate similarity
        score = similarity(query_embedding, embedding)
        similarities.append((id, title, artist, period, description, score))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[5], reverse=True)
    
    conn.close()
    
    # Return top k results
    return similarities[:top_k]

def rag_completion(query, system_prompt="You are a helpful art historian specializing in ukiyo-e Japanese woodblock prints."):
    """Generate a completion based on the query and relevant context."""
    # Search for similar ukiyo-e
    similar_ukiyo_e = search_similar_ukiyo_e(query)
    
    # Build context from search results
    context = "Here is information about some relevant ukiyo-e woodblock prints:\n\n"
    for _, title, artist, period, description, score in similar_ukiyo_e:
        context += f"- Title: {title}\n  Artist: {artist}\n  Period: {period}\n  Description: {description}\n  Relevance: {score:.2f}\n\n"
    
    # Generate completion using OpenAI's API with RAG context
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context information: {context}"},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",  # or any other available model
        messages=messages
    )
    
    return response.choices[0].message.content

def main():
    try:
        # Create the database with ukiyo-e titles
        create_database()
        
        # Example usage
        print("RAG System for Ukiyo-e Art")
        print("==========================")
        
        while True:
            query = input("\nEnter your question about ukiyo-e (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            try:
                print("\nRetrieving relevant information...")
                answer = rag_completion(query)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                print(f"\nError processing query: {e}")
                print("Please try a different question or check your API key.")
    
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("Please check your OpenAI API key and dependencies.")

if __name__ == "__main__":
    main()