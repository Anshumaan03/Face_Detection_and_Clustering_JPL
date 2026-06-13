import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Face Clustering Dashboard",
    layout="wide"
)

st.title("🎭 Face Clustering Dashboard")

# =====================================================
# LOAD JSON
# =====================================================

with open("clustering_results.json", "r") as f:
    data = json.load(f)

best = data["best_params"]

# =====================================================
# METRICS
# =====================================================

st.header("📊 Clustering Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("NMI", f"{best['nmi']:.3f}")
col2.metric("ARI", f"{best['ari']:.3f}")
col3.metric("Purity", f"{best['purity']:.3f}")
col4.metric("Clusters", best["n_clusters"])
col5.metric("Noise %", f"{best['noise_pct']:.2f}")

st.divider()

# =====================================================
# IDENTITY ANALYSIS
# =====================================================

st.header("👥 Identity Analysis")

rows = []

for identity, info in data["per_identity"].items():

    rows.append({
        "Identity": identity,
        "Images": info["total"],
        "Noise Images": info["noise"],
        "Clusters Found": info["n_clusters"]
    })

df = pd.DataFrame(rows)

st.dataframe(
    df.sort_values("Identity"),
    use_container_width=True
)

# =====================================================
# PROBLEM IDENTITIES
# =====================================================

st.header("⚠️ Split Identities")

found_problem = False

for identity, info in data["per_identity"].items():

    if info["n_clusters"] > 1:

        found_problem = True

        st.warning(
            f"{identity} split into {info['n_clusters']} clusters"
        )

if not found_problem:
    st.success("No split identities found")

st.divider()

# =====================================================
# CLUSTER EXPLORER
# =====================================================

st.header("📁 Cluster Explorer")

results_dir = Path("/Users/anshumaansinghrathore/Desktop/Face Clustering/results")

if not results_dir.exists():

    st.error(
        "results folder not found. Run create_cluster_folders.py first."
    )

    st.stop()

cluster_folders = sorted([
    folder.name
    for folder in results_dir.iterdir()
    if folder.is_dir()
])

selected_cluster = st.selectbox(
    "Select Cluster",
    cluster_folders
)

cluster_path = results_dir / selected_cluster

images = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:

    images.extend(cluster_path.glob(ext))

st.write(f"Total Images: {len(images)}")

# =====================================================
# IMAGE GRID
# =====================================================

cols = st.columns(5)

for idx, image_path in enumerate(images):

    try:

        img = Image.open(image_path)

        cols[idx % 5].image(
            img,
            caption=image_path.name,
            use_container_width=True
        )

    except Exception:
        pass

st.divider()

# =====================================================
# CLUSTER SUMMARY
# =====================================================

st.header("📈 Cluster Summary")

cluster_stats = []

for folder in cluster_folders:

    count = len(list((results_dir / folder).glob("*")))

    cluster_stats.append({
        "Cluster": folder,
        "Images": count
    })

cluster_df = pd.DataFrame(cluster_stats)

st.dataframe(
    cluster_df.sort_values(
        "Images",
        ascending=False
    ),
    use_container_width=True
)

st.success("Dashboard Loaded Successfully 🚀")