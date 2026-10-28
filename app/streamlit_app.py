"""Demo UI. Run: streamlit run app/streamlit_app.py  (needs a built index and ANTHROPIC_API_KEY)."""

import streamlit as st
from dotenv import load_dotenv

from biorag.agents.graph import Pipeline, PipelineConfig
from biorag.config import get_settings
from biorag.llm import make_llm
from biorag.retrieval.retriever import HybridRetriever

load_dotenv()
st.set_page_config(page_title="BioRAG", layout="wide")


@st.cache_resource
def pipeline() -> Pipeline:
    s = get_settings()
    return Pipeline(
        make_llm(s, "generator"), make_llm(s, "judge"), HybridRetriever(s),
        PipelineConfig(max_subquestions=s.max_subquestions),
    )


st.title("BioRAG: cited answers from PubMed")
question = st.text_input(
    "Ask a biomedical research question",
    "Does metformin reduce the risk of cancer in patients with type 2 diabetes?",
)

if st.button("Answer", type="primary") and question:
    with st.spinner("Planning, retrieving, answering, verifying..."):
        state = pipeline().run(question)
    answer = state["answer"]
    failed = {v.claim_index for v in state.get("verdicts", []) if not v.supported}

    st.subheader(f"Answer: {answer.decision.upper()}")
    st.write(answer.summary)
    st.markdown("**Claims**")
    for i, claim in enumerate(answer.claims):
        mark = "⚠️ unverified" if i in failed else "✅ verified"
        cites = ", ".join(f"[{c}]" for c in claim.citations)
        st.markdown(f"- {claim.text} {cites} — {mark}")

    with st.expander("Search plan"):
        for q in state.get("subquestions", []):
            st.write(f"- {q}")
    st.markdown("**Evidence**")
    for i, e in enumerate(state["evidence"], 1):
        url = f"https://pubmed.ncbi.nlm.nih.gov/{e.chunk.doc_id}/"
        with st.expander(f"[{i}] PMID {e.chunk.doc_id} (score {e.score:.2f})"):
            st.write(e.chunk.render())
            st.markdown(f"[Open in PubMed]({url})")
