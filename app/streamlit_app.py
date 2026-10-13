import streamlit as st
from rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="GenAI Banking Document Assistant",
    page_icon="🏦",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Banking Assistant")
    st.markdown("**Powered by RAG + FAISS + LLM**")
    st.divider()
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4,
                      help="How many document chunks to retrieve for each question.")
    st.divider()
    st.markdown("**About**")
    st.caption(
        "This assistant answers questions about synthetic banking policies "
        "using Retrieval-Augmented Generation. No real customer data is used."
    )
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Load pipeline ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_pipeline():
    p = RAGPipeline()
    p.load_index()
    return p

try:
    pipeline = get_pipeline()
except FileNotFoundError:
    st.error("⚠️ Index not found. Please run `python -m scripts.ingest` first.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load pipeline: {exc}")
    st.stop()

# ── Chat history ───────────────────────────────────────────────────────────
st.title("🏦 GenAI Banking Document Assistant")
st.caption("Ask anything about banking policies. Answers are grounded in retrieved documents only.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} — chunk {s['chunk']}** (score: `{s['score']:.3f}`)")
                    st.caption(s["text"])

# ── Input ──────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about banking policies…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                result = pipeline.ask(question, top_k)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                with st.expander("📄 Sources", expanded=False):
                    for s in sources:
                        st.markdown(f"**{s['source']} — chunk {s['chunk']}** (score: `{s['score']:.3f}`)")
                        st.caption(s["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except RuntimeError as exc:
                err = f"⚠️ LLM error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as exc:
                err = f"❌ Unexpected error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
