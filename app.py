"""
Streamlit demo for MMSearch-R1.

Launch:
    streamlit run app.py

Shows:
  - Image upload + question input
  - Real-time tool call trace (which searches were made)
  - Final answer with supporting retrieved evidence
"""

import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="MMSearch-R1 Demo",
    page_icon="🔍",
    layout="wide",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MMSearch-R1")
    st.caption("Multi-turn multimodal web search with RL-trained reasoning")

    model_path = st.text_input(
        "Model path",
        value=os.environ.get("MMSEARCH_MODEL", "lmms-lab/MMSearch-R1-7B"),
    )
    max_new_tokens = st.slider("Max new tokens per turn", 64, 1024, 512, 64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    image_search_limit = st.slider("Image search limit", 0, 3, 1)
    text_search_limit = st.slider("Text search limit", 0, 5, 2)
    max_turns = st.slider("Max turns", 1, 5, 3)

    st.divider()
    st.caption("API keys loaded from .env")
    serpapi_ok = bool(os.environ.get("SERPAPI_KEY"))
    st.markdown(f"SerpAPI: {'✅' if serpapi_ok else '❌ not set'}")
    jina_ok = bool(os.environ.get("JINA_API_KEY"))
    st.markdown(f"Jina: {'✅' if jina_ok else '⚠️ optional'}")
    dash_ok = bool(os.environ.get("DASHSCOPE_API_KEY"))
    st.markdown(f"DashScope: {'✅' if dash_ok else '⚠️ optional (fallback: truncation)'}")


# ─── Model loading (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading MMSearch-R1 model…")
def get_engine(path: str):
    from inference.engine import MMSearchEngine
    return MMSearchEngine(model_id=path)


# ─── Main UI ─────────────────────────────────────────────────────────────────

st.title("🔍 MMSearch-R1 — Multimodal Web Search")
st.markdown(
    "Upload an image (optional) and ask a question. "
    "The model will autonomously search the web when needed."
)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)
    else:
        image = None

with col2:
    question = st.text_area(
        "Question",
        placeholder="What is shown in this image? Who created it? When?",
        height=100,
    )
    run_btn = st.button("Search & Answer", type="primary", disabled=not question.strip())

if run_btn and question.strip():
    from inference.engine import GenerationConfig

    engine = get_engine(model_path)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        image_search_limit=image_search_limit,
        text_search_limit=text_search_limit,
        max_turns=max_turns,
    )

    with st.spinner("Reasoning and searching…"):
        result = engine.run(question=question, image=image, config=config, verbose=False)

    st.divider()

    # Answer
    st.subheader("Answer")
    st.success(result["answer"] if result["answer"] else "_No answer extracted_")

    # Tool call trace
    if result["tool_calls"]:
        st.subheader(f"Search trace ({len(result['tool_calls'])} tool call{'s' if len(result['tool_calls']) != 1 else ''})")
        for i, call in enumerate(result["tool_calls"], 1):
            if call["type"] == "image_search":
                st.markdown(f"**{i}.** 🖼 Image search → `{call['url']}`")
            else:
                st.markdown(f"**{i}.** 🔤 Text search → `{call['query']}`")
    else:
        st.info("No web searches were needed — the model answered from memory.")

    # Full reasoning
    with st.expander("Full reasoning trace"):
        st.text(result["full_response"])

    st.caption(f"Completed in {result['turns']} turn{'s' if result['turns'] != 1 else ''}")
