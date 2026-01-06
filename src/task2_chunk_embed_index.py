import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

# -------------------------
# PARAMETERS
# -------------------------
SAMPLE_SIZE = 12000  # stratified sample size
CHUNK_SIZE = 500     # max characters per chunk
CHUNK_OVERLAP = 50   # overlap between chunks
VECTOR_STORE_DIR = "vector_store"

# -------------------------
# LOAD CLEANED DATA
# -------------------------
df = pd.read_csv("data/processed/filtered_complaints.csv")

# Drop any rows with missing narratives
df = df[df["cleaned_narrative"].notna()]

# Ensure proportional representation for product categories
products = df['product'].unique()
sampled_df_list = []

for product in products:
    prod_df = df[df['product'] == product]
    n_sample = int(SAMPLE_SIZE * len(prod_df)/len(df))
    sampled_df_list.append(prod_df.sample(n=n_sample, random_state=42))

sampled_df = pd.concat(sampled_df_list).reset_index(drop=True)
print(f"Sampled dataset size: {len(sampled_df)}")

# -------------------------
# CHUNKING
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = []
metadata = []

for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Chunking"):
    splits = text_splitter.split_text(row["cleaned_narrative"])
    chunks.extend(splits)
    # attach metadata for each chunk
    metadata.extend([{
        "complaint_id": row["complaint_id"],
        "product": row["product"]
    } for _ in splits])

print(f"Total chunks created: {len(chunks)}")

# -------------------------
# EMBEDDING
# -------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = []
for chunk in tqdm(chunks, desc="Generating embeddings"):
    emb = model.encode(chunk)
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# -------------------------
# FAISS INDEX
# -------------------------
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "faiss_index.index"))

with open(os.path.join(VECTOR_STORE_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print(f"FAISS index and metadata saved in {VECTOR_STORE_DIR}/")
