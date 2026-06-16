"""
Combined Face Clustering Dashboard
Compares HDBSCAN, K-Means, and DBSCAN side by side.

HOW TO RUN:
  python hdbscan_pipeline.py
  python kmeans_pipeline.py
  python dbscan_pipeline.py
  streamlit run app_combined.py
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Face Clustering — Algorithm Comparison", layout="wide")
st.title("🎭 Face Clustering — Algorithm Comparison")
st.caption("HDBSCAN vs K-Means vs DBSCAN on ArcFace embeddings (aligned pipeline)")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _format_params(bp, algo):
    if algo == "HDBSCAN":
        return f"min_cls={bp.get('min_cluster_size','?')}, min_smp={bp.get('min_samples','?')}"
    elif algo == "K-Means":
        return f"K={bp.get('k','?')}"
    elif algo == "DBSCAN":
        return f"eps={bp.get('eps','?')}, min_smp={bp.get('min_samples','?')}"
    return ""


# ─────────────────────────────────────────────────────────────
# Load JSON results
# ─────────────────────────────────────────────────────────────
RESULTS = {
    "HDBSCAN": "clustering_results.json",
    "K-Means": "clustering_results_kmeans.json",
    "DBSCAN":  "clustering_results_dbscan.json",
}

RESULT_DIRS = {
    "HDBSCAN": Path("results"),
    "K-Means": Path("results_kmeans"),
    "DBSCAN":  Path("results_dbscan"),
}

loaded = {}
for algo, path in RESULTS.items():
    if Path(path).exists():
        with open(path) as f:
            loaded[algo] = json.load(f)
    else:
        loaded[algo] = None

available = [a for a, d in loaded.items() if d is not None]
missing   = [a for a, d in loaded.items() if d is None]

if missing:
    for m in missing:
        st.warning(f"⚠️  {m} results not found — run the corresponding pipeline script first.")

if not available:
    st.error("No clustering results found. Run the pipeline scripts first.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Section 1 — Side-by-side metric comparison
# ─────────────────────────────────────────────────────────────
st.header("📊 Algorithm Comparison")

metric_rows = []
for algo in available:
    bp = loaded[algo]["best_params"]
    metric_rows.append({
        "Algorithm":   algo,
        "NMI":         round(bp["nmi"], 4),
        "ARI":         round(bp["ari"], 4),
        "Purity":      round(bp["purity"], 4),
        "Clusters":    bp["n_clusters"],
        "Noise %":     round(bp["noise_pct"], 2),
        "Best Params": _format_params(bp, algo),
    })

df_metrics = pd.DataFrame(metric_rows).set_index("Algorithm")
st.dataframe(df_metrics, use_container_width=True)

# Highlight best per metric
col1, col2, col3 = st.columns(3)
for col, metric in zip([col1, col2, col3], ["NMI", "ARI", "Purity"]):
    best_algo = df_metrics[metric].idxmax()
    best_val  = df_metrics[metric].max()
    col.metric(f"Best {metric}", f"{best_val:.4f}", best_algo)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 2 — Bar chart comparison
# ─────────────────────────────────────────────────────────────
st.header("📈 Metric Chart")

chart_df = df_metrics[["NMI", "ARI", "Purity"]].reset_index()
chart_df = chart_df.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
st.bar_chart(chart_df.pivot(index="Algorithm", columns="Metric", values="Score"))

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 3 — Per-identity breakdown
# ─────────────────────────────────────────────────────────────
st.header("👥 Per-Identity Breakdown")

selected_algo = st.selectbox("Select algorithm", available, key="identity_algo")
per_id = loaded[selected_algo].get("per_identity", {})

if per_id:
    id_rows = []
    for identity, info in per_id.items():
        n_cls = info["n_clusters"]
        status = "✅ Clean" if n_cls == 1 else ("⚠️ Split" if n_cls > 1 else "❌ All Noise")
        id_rows.append({
            "Identity":       identity,
            "Total Images":   info["total"],
            "Noise":          info["noise"],
            "Clusters Found": n_cls,
            "Status":         status,
        })
    st.dataframe(pd.DataFrame(id_rows).sort_values("Identity"), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 4 — Cluster Explorer
# ─────────────────────────────────────────────────────────────
st.header("📁 Cluster Explorer")

explorer_algo = st.selectbox("Algorithm", available, key="explorer_algo")
result_dir    = RESULT_DIRS[explorer_algo]

if not result_dir.exists():
    st.warning(f"Cluster folders for {explorer_algo} not found at `{result_dir}`.")
else:
    cluster_folders = sorted([f.name for f in result_dir.iterdir() if f.is_dir()])
    if not cluster_folders:
        st.info("No cluster folders found.")
    else:
        selected_cluster = st.selectbox("Select Cluster", cluster_folders, key="cluster_sel")
        cluster_path     = result_dir / selected_cluster

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend(cluster_path.glob(ext))
        images = sorted(images)

        st.write(f"**{len(images)} images** in `{selected_cluster}`")

        cols = st.columns(5)
        for idx, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                cols[idx % 5].image(img, caption=img_path.name, use_container_width=True)
            except Exception:
                pass

st.divider()

# ─────────────────────────────────────────────────────────────
# Section 5 — Full hyperparameter sweep table
# ─────────────────────────────────────────────────────────────
st.header("🔬 Full Hyperparameter Sweep")

sweep_algo = st.selectbox("Algorithm", available, key="sweep_algo")
all_res    = loaded[sweep_algo].get("all_results", [])

if all_res:
    st.dataframe(pd.DataFrame(all_res), use_container_width=True)
else:
    st.info("No sweep data available for this algorithm.")

st.success("Dashboard loaded ✅")
