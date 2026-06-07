import os

local_path = "gte-small-local"

if os.path.exists(local_path) and os.path.isdir(local_path):
    print(f"✅ Model already exists at '{local_path}'. Skipping download.")
else:
    from sentence_transformers import SentenceTransformer
    print("🔽 Start downloading embedding model...")
    model = SentenceTransformer("thenlper/gte-small")
    model.save(local_path)
    print("✅ Model saved into 'gte-small-local'")
