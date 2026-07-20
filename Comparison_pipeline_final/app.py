import os
import glob

import streamlit as st

import config
from clustering import cluster_all_models, metrics_summary_table

st.set_page_config(page_title="Face Embedding Comparison", layout="wide")
st.title("Face Embedding Model Comparison")
st.caption("ArcFace · dlib-ResNet · FaceNet (VGGFace2) · SigLIP2 — common detector, per-model alignment, identical HDBSCAN")

run_label = st.sidebar.text_input("Run label", value="default")

if st.sidebar.button("Re-run clustering (uses existing DB embeddings)"):
    with st.spinner("Running HDBSCAN for all models..."):
        results = cluster_all_models(run_label=run_label)
        st.session_state["results"] = results
    st.sidebar.warning("Re-clustering just rebuilt cluster_results from scratch for this run_label.")

if "results" not in st.session_state:
    with st.spinner("Loading clustering results..."):
        st.session_state["results"] = cluster_all_models(run_label=run_label)

results = st.session_state["results"]

# ===========================================================================
# Metrics + cluster browser
# ===========================================================================
st.subheader("Metrics comparison")
table = metrics_summary_table(results)
st.dataframe(table.style.format("{:.3f}", subset=[c for c in table.columns if table[c].dtype != "int64"]))
st.bar_chart(table["ari"])

st.subheader("Cluster browser")
col1, col2 = st.columns(2)
with col1:
    model_choice_browse = st.selectbox("Model", list(config.MODEL_INPUT_SPECS.keys()), key="browse_model")

cluster_dir = os.path.join(config.CLUSTERS_ROOT, run_label, model_choice_browse)

if not os.path.isdir(cluster_dir):
    st.warning(f"No cluster folder found at {cluster_dir}. Run the pipeline / clustering first.")
else:
    cluster_folders = sorted(
        os.listdir(cluster_dir),
        key=lambda x: (x != "noise", x)
    )
    with col2:
        cluster_choice = st.selectbox("Cluster", cluster_folders, key="browse_cluster")

    chosen_dir = os.path.join(cluster_dir, cluster_choice)
    image_paths = sorted(glob.glob(os.path.join(chosen_dir, "*")))

    st.write(f"**{len(image_paths)} images** in `{model_choice_browse}/{cluster_choice}`")

    identities_in_cluster = sorted({os.path.basename(p).split("__")[0] for p in image_paths})
    if len(identities_in_cluster) > 1:
        st.error(f"Mixed identities in this cluster: {identities_in_cluster}")
    else:
        st.success(f"Pure cluster: {identities_in_cluster[0] if identities_in_cluster else 'n/a'}")

    n_cols = 6
    cols = st.columns(n_cols)
    for i, img_path in enumerate(image_paths):
        with cols[i % n_cols]:
            st.image(img_path, use_container_width=True, caption=os.path.basename(img_path).split("__")[0])
