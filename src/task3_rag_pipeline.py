import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

VECTOR_STORE_PATH = "vector_store/index.pkl"

# -----------------------------
# Initialize embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load LLM using HuggingFace pipeline
# -----------------------------
generator_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    max_new_tokens=512,  # better than max_length
    pad_token_id=11,
)
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# -----------------------------
# Load or rebuild vector store
# -----------------------------
print("Loading FAISS vector store...")
try:
    with open(VECTOR_STORE_PATH, "rb") as f:
        docs = pickle.load(f)

    # Convert each dict into a clean string for embeddings
    texts = []
    for doc in docs:
        text = " | ".join([f"{k}: {v}" for k, v in doc.items()])
        texts.append(text)

    # Load FAISS or rebuild if needed
    vector_store = FAISS.from_texts(texts, embedding_model)
    print("Vector store loaded successfully.")
except Exception as e:
    print("Failed to load vector store. Rebuild your FAISS index with all complaints.")
    print("Error:", str(e))
    exit(1)

# -----------------------------
# Query RAG function
# -----------------------------
def query_rag(question, k=5):
    """
    Query the RAG system:
    - Retrieve top-k similar complaint excerpts
    - Build a prompt for the LLM
    - Return generated answer
    """
    # Retrieve top k relevant docs
    docs_with_scores = vector_store.similarity_search_with_score(question, k=k)

    # Extract and clean retrieved texts for prompt
    retrieved_texts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        text = str(doc)
        # Remove page_content='...' and metadata={}
        if "page_content=" in text:
            text = text.replace("page_content='", "").replace("' metadata={}", "")
        retrieved_texts.append(text)

    # Show retrieved context for transparency
    print("\nRetrieved context:")
    for i, t in enumerate(retrieved_texts, 1):
        print(f"{i}. {t}")

    context = "\n".join(retrieved_texts)

    # Build prompt for LLM
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use ONLY the following retrieved complaint excerpts to answer the question. 
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:
"""

    # Generate answer
    answer = llm(prompt)

    # HuggingFacePipeline sometimes returns a list of dicts
    if isinstance(answer, list) and isinstance(answer[0], dict) and "generated_text" in answer[0]:
        return answer[0]["generated_text"].strip()

    # Otherwise, return as string
    return str(answer).strip()

# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    questions = [
        "What issues did customers have with credit cards?",
        "How many complaints mention late fees?",
        "Summarize common problems with mortgage products.",
        "Which products have the highest number of complaints?",
        "Are there complaints about auto loans?"
    ]

    for q in questions:
        print("\n" + "=" * 50)
        print(f"Question: {q}")
        print("Answer:", query_rag(q))
        print("=" * 50)
