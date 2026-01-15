import gradio as gr
import pickle

from transformers import pipeline

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline



# === Load models ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_length=512)
llm = HuggingFacePipeline(pipeline=generator)

# === Load FAISS index ===
VECTOR_STORE_PATH = "vector_store/index.pkl"
with open(VECTOR_STORE_PATH, "rb") as f:
    docs = pickle.load(f)
texts = [" | ".join([f"{k}: {v}" for k, v in d.items()]) for d in docs]
vector_store = FAISS.from_texts(texts, embedding_model)

# === Query function ===
def rag_chat(question, k=5):
    docs_with_scores = vector_store.similarity_search_with_score(question, k=k)
    retrieved = [str(doc[0]) for doc in docs_with_scores]
    context = "\n".join(retrieved)
    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Use the following context to answer the question. If information is missing, say so.

Context:
{context}

Question: {question}

Answer:
"""
    answer = llm(prompt)
    answer_text = answer[0]["generated_text"] if isinstance(answer, list) else str(answer)
    return answer_text, context

# === Gradio UI ===
with gr.Blocks(title="CrediTrust Complaint RAG Chatbot") as demo:
    gr.Markdown("## 🧠 CrediTrust Complaint Chatbot\nAsk a question about customer complaints.")
    question = gr.Textbox(label="Enter your question")
    answer = gr.Textbox(label="Answer", interactive=False)
    sources = gr.Textbox(label="Retrieved Context", interactive=False)
    btn = gr.Button("Ask")

    btn.click(fn=rag_chat, inputs=question, outputs=[answer, sources])

# === Launch ===
if __name__ == "__main__":
    demo.launch(share=True)
