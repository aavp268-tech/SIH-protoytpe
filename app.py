import streamlit as st
from recommender import Recommender
from extract import pdf_to_text

st.set_page_config(page_title="IS Recommender", layout="wide")
st.title("AI-Powered Indian Standards Recommender (SIH PS 26108)")
@st.cache_resource
def load(): return Recommender()
rec = load()

up = st.file_uploader("Upload tender / specification PDF (optional)", type="pdf")
default = "Supply of TMT reinforcement bars Fe 500 for RCC building construction"
q = st.text_area("Procurement requirement", pdf_to_text(up)[:3000] if up else default, height=120)
c1, c2 = st.columns(2)
k = c1.slider("Top-K", 1, 10, 5)
cfg = c2.radio("Retrieval configuration", ["C: Hybrid+Rerank", "A: BM25", "B: Semantic"], horizontal=True)
from evaluate import CFG
if st.button("Recommend standards", type="primary") and q.strip():
    res, req = rec.search(q, k, **CFG[cfg])
    with st.expander("Extracted requirement", expanded=True): st.json(req)
    for r in res:
        with st.expander(f"{r['is_number']}:{r['year']} — {r['title']}  (score {r['score']:.2f})", expanded=True):
            st.write(r["explanation"])
            st.caption(f"Class: {r['classification']}  |  Version: {r['version']}")
            st.write("Knowledge-graph links:", ", ".join(f"{v} ({rel})" for v, rel in r["graph"]) or "—")
