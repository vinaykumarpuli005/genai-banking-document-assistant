import streamlit as st
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="GenAI Banking Document Assistant", page_icon="🏦", layout="wide")
st.title("🏦 GenAI Banking Document Assistant")
st.caption("RAG + Prompt Engineering + Embeddings + FAISS + REST API")

@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    pipeline.load_index()
    return pipeline

try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(str(exc))
    st.stop()

question = st.text_input("Ask a question about the banking policies",
                         placeholder="Example: What is the maximum daily ATM withdrawal limit?")
top_k = st.slider("Retrieved chunks", 1, 8, 4)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = pipeline.ask(question, top_k)
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Retrieved Sources")
            for source in result["sources"]:
                st.write(f"**{source['source']} — chunk {source['chunk']}** (similarity: {source['score']:.3f})")
                st.caption(source["text"])
        except Exception as exc:
            st.error(str(exc))
