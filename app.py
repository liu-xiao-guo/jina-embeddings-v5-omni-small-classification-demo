import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import datasets
import eis
import store

st.set_page_config(page_title="Jina v5 Omni Small – Classification demo", page_icon="🧭", layout="wide")
MEDIA_EXT = {"image": ["png", "jpg", "jpeg", "webp", "gif", "bmp"], "video": ["mp4", "mov", "webm", "mkv", "avi"],
             "audio": ["wav", "mp3", "flac", "ogg", "m4a"]}
ICON = {"text": "📝", "image": "🖼️", "video": "🎬", "audio": "🔊"}

st.title("🧭 jina-embeddings-v5-omni-small-classification")
st.caption("One shared 1024-d vector space for text, images, video and audio — served by Elastic Inference Service "
           f"(`{eis.OMNI_ID}`). Index with text, query with any modality, no reindexing.")


def show_media(item: dict, width=180):
    m = item["modality"]
    if m == "image":
        st.image(item["path"], width=width)
    elif m == "video":
        st.video(item["path"])
    elif m == "audio":
        st.audio(item["path"])


def get_query_vector(key: str, dims: int):
    """Query picker for any modality. Returns (vector, preview_fn) or (None, None)."""
    modality = st.radio("Query modality", ["text", "image", "video", "audio"], horizontal=True, key=f"{key}-mod")
    if modality == "text":
        q = st.text_input("Query text", "a red circle", key=f"{key}-text")
        if q:
            return modality, eis.embed_texts([q], dims, prefix=st.session_state.get("prefix", True))[0], lambda: st.write(f"“{q}”")
        return None, None, None
    samples = [i for i in datasets.load_manifest() if i["modality"] == modality]
    src = st.radio("Source", ["Sample", "Upload"], horizontal=True, key=f"{key}-src")
    if src == "Sample" and samples:
        pick = st.selectbox("Sample", samples, format_func=lambda i: i["title"], key=f"{key}-sample")
        path = Path(pick["path"])
        data, name = path.read_bytes(), path.name
    else:
        up = st.file_uploader(f"Upload {modality}", type=MEDIA_EXT[modality], key=f"{key}-up")
        if not up:
            return None, None, None
        data, name = up.getvalue(), up.name

    def preview():
        {"image": st.image, "video": st.video, "audio": st.audio}[modality](data) if modality != "image" else st.image(data, width=220)

    with st.spinner(f"Embedding {modality} via EIS…"):
        vec = eis.embed_media(modality, data, name, dims)
    return modality, vec, preview


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    st.session_state["prefix"] = st.checkbox('Prepend "Document: " to text', True,
                                             help="Required by the model card; raw text is off-distribution.")
    st.divider()
    try:
        counts = store.count_by_modality()
        st.success("Elasticsearch connected")
        st.write(counts or "Index not built yet")
    except Exception as e:
        st.error(f"Elasticsearch error: {e}")

tab_data, tab_search, tab_cls, tab_align = st.tabs(["1 · Datasets & index", "2 · Cross-modal search", "3 · Zero-shot classification", "4 · Shared-space check"])

# ------------------------------------------------------------------ tab 1
with tab_data:
    c1, c2 = st.columns(2)
    if c1.button("① Generate sample datasets", width="stretch"):
        with st.spinner("Generating images, videos, audio and text…"):
            datasets.generate_all()
        st.toast("Datasets generated")
    items = datasets.load_manifest()
    if c2.button("② Embed via EIS & index into Elasticsearch", type="primary", width="stretch", disabled=not items):
        bar = st.progress(0.0)
        n = store.index_items(items, lambda p, msg: bar.progress(p, msg))
        bar.progress(1.0, f"Indexed {n} documents")
        st.rerun()
    if not items:
        st.info("Click **Generate sample datasets** to start.")
    for m in ("text", "image", "video", "audio"):
        sub = [i for i in items if i["modality"] == m]
        if not sub:
            continue
        with st.expander(f"{ICON[m]} {m} — {len(sub)} items", expanded=m in ("text", "image")):
            if m == "text":
                st.table(pd.DataFrame(sub).set_index("id")[["title", "caption"]])
            else:
                cols = st.columns(4)
                for i, it in enumerate(sub):
                    with cols[i % 4]:
                        show_media(it)
                        st.caption(f"**{it['title']}**  \n{it['caption']}")

# ------------------------------------------------------------------ tab 2
with tab_search:
    if not store.index_exists():
        st.info("Build the index in tab 1 first.")
    else:
        left, right = st.columns([1, 2])
        with left:
            modality, vec, preview = get_query_vector("search", eis.DIMS)
            targets = st.multiselect("Search in", ["text", "image", "video", "audio"], default=["text", "image", "video", "audio"])
            k = st.slider("Results per modality", 1, 8, 3)
            if vec is not None:
                st.markdown("**Query**")
                preview()
        with right:
            if vec is None:
                st.stop()
            st.subheader(f"{ICON[modality]} {modality} query → results")
            st.caption("Top-k per target modality (cross-modal cosine scores are naturally lower than same-modality ones, so each modality is ranked on its own).")
            results = store.search(vec, targets, k)
            for m, hits in results.items():
                st.markdown(f"##### {ICON[m]} {m}")
                cols = st.columns(max(len(hits), 1))
                for col, h in zip(cols, hits):
                    with col:
                        show_media(h, width=140)
                        st.markdown(f"**{h['title']}**")
                        st.caption(h["caption"])
                        st.progress(float(np.clip(h["score"], 0, 1)), text=f"score {h['score']:.3f}")

# ------------------------------------------------------------------ tab 3
with tab_cls:
    st.markdown("Zero-shot classification: embed candidate **labels as text** and assign any modality to the nearest label.")
    default_labels = "a red shape\na blue shape\na green shape\na striped or checkered pattern\na night sky\na loud alarm or siren\nmusic\nnoise or static\nsnow or rain"
    labels = [l.strip() for l in st.text_area("Labels (one per line)", default_labels, height=200).splitlines() if l.strip()]
    temp = st.slider("Softmax temperature", 0.01, 0.2, 0.05, 0.01)
    dims = st.select_slider("Matryoshka dimensions", [32, 64, 128, 256, 512, 768, 1024], value=1024,
                            help="Query and labels are both truncated and re-normalized to this size.")
    c_left, c_right = st.columns([1, 1])
    with c_left:
        modality, vec, preview = get_query_vector("cls", dims)
        if vec is not None:
            preview()
    with c_right:
        if vec is not None and labels:
            L = eis.embed_texts(labels, dims, prefix=st.session_state["prefix"])
            sims = L @ vec
            p = np.exp((sims - sims.max()) / temp)
            p /= p.sum()
            df = pd.DataFrame({"label": labels, "probability": p, "cosine": sims}).sort_values("probability", ascending=False)
            st.success(f"Predicted: **{df.iloc[0]['label']}** ({df.iloc[0]['probability']:.0%})")
            st.bar_chart(df.set_index("label")["probability"], horizontal=True)
            st.dataframe(df, hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 4
with tab_align:
    st.markdown(f"Text embedded with the **omni** endpoint (`{eis.OMNI_ID}`) vs the **text-only** endpoint "
                f"(`{eis.TEXT_ID}`). If the spaces are aligned, cosine similarity ≈ 1.0 — so a text index built with "
                "the text-only model is directly queryable with image/video/audio from the omni model.")
    txt = st.text_area("Texts", "A cute cat sitting on a mat.\nThe stock market closed higher today.", height=100)
    if st.button("Compare"):
        ts = [t for t in txt.splitlines() if t.strip()]
        try:
            a = eis.embed_texts(ts, eis.DIMS)
            b = eis.embed_texts(ts, eis.DIMS, endpoint=eis.TEXT_ID)
            st.dataframe(pd.DataFrame({"text": ts, "cosine(omni, text-only)": (a * b).sum(1)}), hide_index=True, width="stretch")
        except Exception as e:
            st.error(e)
