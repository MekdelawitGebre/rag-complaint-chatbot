import gradio as gr
import pickle
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# === Load models ===
print("Loading models...")

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM (Falcon)
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_length=512)
llm = HuggingFacePipeline(pipeline=generator)

# === Load FAISS vector store ===
VECTOR_STORE_DIR = "vector_store"
INDEX_FILE = f"{VECTOR_STORE_DIR}/index.faiss"
META_FILE = f"{VECTOR_STORE_DIR}/index.pkl"

print("Loading FAISS vector store...")
try:
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully.")
except Exception as e:
    print("❌ Failed to load vector store using FAISS.load_local:", e)
    # Fallback: try loading raw documents from the pickle and build the FAISS index in-memory
    try:
        with open(META_FILE, "rb") as mf:
            docs = pickle.load(mf)
        # Convert docs (list of dicts) into text strings
        texts = [" | ".join([f"{k}: {v}" for k, v in d.items()]) for d in docs]
        print(f"Building FAISS vector store from {len(texts)} documents (this may take a while)...")
        vector_store = FAISS.from_texts(texts, embedding_model)
        print("Vector store built from pickle successfully.")
    except Exception as e2:
        print("❌ Fallback failed — could not build vector store from pickle:", e2)
        vector_store = None


# === RAG pipeline function ===
def rag_chat(question):
    """Retrieve relevant docs and generate an answer."""
    if not vector_store:
        return "⚠️ Error: Vector store not loaded properly. Please check your FAISS index files."

    # Retrieve top-k most relevant chunks
    try:
        docs_with_scores = vector_store.similarity_search_with_score(question, k=5)
        retrieved_texts = []
        for doc, score in docs_with_scores:
            if hasattr(doc, "page_content"):
                retrieved_texts.append(doc.page_content)
            elif isinstance(doc, dict):
                retrieved_texts.append(" | ".join([f"{k}: {v}" for k, v in doc.items()]))
            else:
                retrieved_texts.append(str(doc))
    except Exception as e:
        return f"Error retrieving context: {e}"

    context = "\n".join(retrieved_texts)
    prompt = f"""
You are a financial analyst assistant for CrediTrust. 
Use only the following complaint excerpts to answer the user's question.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
"""

    try:
        # Use invoke (modern LangChain API)
        response = llm.invoke(prompt)
        answer_text = response if isinstance(response, str) else str(response)
        return f"### 🧠 Answer:\n{answer_text}\n\n---\n### 📄 Retrieved Context:\n{context}"
    except Exception as e:
        return f"Error generating answer: {e}"


# === Gradio App ===
with gr.Blocks(title="CrediTrust Complaint Chatbot") as demo:
    gr.Markdown("# 💬 CrediTrust Complaint Chatbot")
    gr.Markdown("Ask questions about financial complaints and get contextual answers with sources.")

    question = gr.Textbox(label="Enter your question", placeholder="e.g. What issues did customers have with credit cards?")
    output = gr.Markdown(label="Response")

    ask_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear")

    ask_btn.click(fn=rag_chat, inputs=question, outputs=output)
    clear_btn.click(fn=lambda: "", outputs=output)

demo.launch()

