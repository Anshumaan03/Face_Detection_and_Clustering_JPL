"""
dashboard.py — Face Clustering Pipeline Visualisation
======================================================
Run with: streamlit run dashboard.py

Pages:
  1. Overview       — dataset summary + embedding quality
  2. Cluster Map    — 2D UMAP projection of all embeddings
  3. Per-Identity   — per-identity cluster breakdown
  4. Frontal vs Profile — ARI comparison
  5. Recommendation — interactive merge UI
"""

import streamlit as st
import mysql.connector
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================================================
# ⚙️  CONFIG
# ================================================================
DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db_v2"
}

HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES      = 1

st.set_page_config(
    page_title  = "Face Clustering Dashboard",
    page_icon   = "🧑",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ================================================================
# 🎨  CUSTOM CSS
# ================================================================
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #cba6f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a6adc8;
        margin-top: 4px;
    }
    .metric-good  { color: #a6e3a1; }
    .metric-mid   { color: #f9e2af; }
    .metric-bad   { color: #f38ba8; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #cdd6f4;
        border-bottom: 1px solid #313244;
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 🔌  DATA LOADING — cached so DB is only hit once per session
# ================================================================
@st.cache_data(ttl=60)
def load_data():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, file_name, identity_label, embedding_json,
               is_profile, eye_distance, detector_source, det_confidence
        FROM   face_embeddings_v2
        ORDER  BY identity_label, file_name
    """)
    rows = cursor.fetchall()

    cursor.execute("""
        SELECT cluster_id, identity_label, centroid_json, member_count
        FROM   cluster_centroids
        ORDER  BY cluster_id
    """)
    centroid_rows = cursor.fetchall()

    cursor.execute("""
        SELECT * FROM merge_decisions ORDER BY decided_at DESC LIMIT 20
    """)
    merge_rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rows, centroid_rows, merge_rows

@st.cache_data(ttl=60)
def compute_clusters(embeddings_json, min_cls, min_smp):
    X = np.array([json.loads(e) for e in embeddings_json], dtype=np.float32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size         = min_cls,
        min_samples              = min_smp,
        metric                   = "euclidean",
        cluster_selection_method = "eom",
        prediction_data          = True
    )
    labels = clusterer.fit_predict(X)
    return X, labels

@st.cache_data(ttl=120)
def compute_umap(X_json):
    """2D UMAP projection for visualisation."""
    try:
        import umap
        X = np.array(json.loads(X_json), dtype=np.float32)
        reducer = umap.UMAP(
            n_components = 2,
            metric       = "cosine",
            n_neighbors  = 15,
            min_dist     = 0.1,
            random_state = 42
        )
        return reducer.fit_transform(X)
    except ImportError:
        return None

# ================================================================
# 🚀  MAIN APP
# ================================================================
def main():
    # ── Sidebar ───────────────────────────────────────────────────
    st.sidebar.image(
        "https://img.icons8.com/color/96/face-id.png",
        width=80
    )
    st.sidebar.title("Face Clustering")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview",
         "🗺️ Cluster Map",
         "👤 Per-Identity Analysis",
         "📐 Frontal vs Profile",
         "🔀 Recommendation System",
         "📋 Merge History"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**HDBSCAN Config**")
    min_cls = st.sidebar.slider("min_cluster_size", 2, 15, HDBSCAN_MIN_CLUSTER_SIZE)
    min_smp = st.sidebar.slider("min_samples",      1, 5,  HDBSCAN_MIN_SAMPLES)

    # ── Load data ─────────────────────────────────────────────────
    with st.spinner("Loading data from MySQL ..."):
        rows, centroid_rows, merge_rows = load_data()

    if not rows:
        st.error("No data found in face_embeddings_v2. Run pipeline_v2.py first.")
        return

    # ── Parse into dataframe ──────────────────────────────────────
    df = pd.DataFrame(rows)
    df["is_profile"] = df["is_profile"].astype(bool)

    embeddings_json = df["embedding_json"].tolist()
    X, labels       = compute_clusters(embeddings_json, min_cls, min_smp)

    df["cluster"]    = labels
    df["cluster_str"]= df["cluster"].apply(
        lambda x: f"Noise" if x == -1 else f"Cluster {x}"
    )

    # Encode ground truth
    le            = LabelEncoder()
    ground_truths = le.fit_transform(df["identity_label"].tolist())
    n_identities  = len(le.classes_)
    total         = len(df)
    n_clusters    = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise       = int(np.sum(labels == -1))

    # Metrics
    nmi = normalized_mutual_info_score(ground_truths, labels)
    ari = adjusted_rand_score(ground_truths, labels)
    try:
        non_noise = labels != -1
        sil = silhouette_score(
            X[non_noise], labels[non_noise], metric="cosine"
        ) if np.sum(non_noise) > 1 else 0.0
    except Exception:
        sil = 0.0

    sil_raw = silhouette_score(X, ground_truths, metric="cosine")

    # ================================================================
    # PAGE 1 — OVERVIEW
    # ================================================================
    if page == "📊 Overview":
        st.title("📊 Face Clustering Dashboard")
        st.markdown(f"**Dataset:** {n_identities} identities · "
                    f"{total} embeddings · "
                    f"HDBSCAN(min_cls={min_cls}, min_smp={min_smp})")
        st.markdown("---")

        # ── Top metrics ───────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        def color_class(val, good, mid):
            if val >= good: return "metric-good"
            if val >= mid:  return "metric-mid"
            return "metric-bad"

        metrics = [
            (c1, f"{nmi:.4f}",  "NMI",             color_class(nmi, 0.7, 0.5)),
            (c2, f"{ari:.4f}",  "ARI",             color_class(ari, 0.6, 0.4)),
            (c3, f"{sil:.4f}",  "Silhouette",      color_class(sil, 0.4, 0.2)),
            (c4, f"{sil_raw:.4f}", "Raw Sil.",      color_class(sil_raw, 0.3, 0.1)),
            (c5, f"{n_clusters}", "Clusters Found", "metric-mid"),
            (c6, f"{n_noise/total*100:.1f}%", "Noise %",
             color_class(1 - n_noise/total, 0.85, 0.75)),
        ]
        for col, val, label, cls in metrics:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Dataset composition ───────────────────────────────────
        st.markdown('<div class="section-header">Dataset Composition</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            identity_counts = df.groupby("identity_label").size().reset_index(name="count")
            identity_counts.columns = ["Identity", "Embeddings"]
            fig = px.bar(
                identity_counts, x="Identity", y="Embeddings",
                color="Identity",
                title="Embeddings per Identity",
                template="plotly_dark"
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            frontal_ct = int((~df["is_profile"]).sum())
            profile_ct = int(df["is_profile"].sum())
            fig2 = go.Figure(go.Pie(
                labels=["Frontal", "Profile"],
                values=[frontal_ct, profile_ct],
                marker_colors=["#a6e3a1", "#f38ba8"],
                hole=0.4
            ))
            fig2.update_layout(
                title="Frontal vs Profile Faces",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Detector source breakdown ─────────────────────────────
        st.markdown('<div class="section-header">Detector Source</div>',
                    unsafe_allow_html=True)
        det_counts = df["detector_source"].value_counts().reset_index()
        det_counts.columns = ["Detector", "Count"]
        fig3 = px.bar(
            det_counts, x="Detector", y="Count",
            color="Detector",
            title="Faces detected by YOLO vs RetinaFace",
            template="plotly_dark",
            color_discrete_map={"yolo": "#89b4fa", "retina": "#cba6f7"}
        )
        fig3.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig3, use_container_width=True)

        # ── Confidence distribution ───────────────────────────────
        st.markdown('<div class="section-header">Detection Confidence Distribution</div>',
                    unsafe_allow_html=True)
        fig4 = px.histogram(
            df, x="det_confidence", color="detector_source",
            nbins=30, template="plotly_dark",
            title="Detection Confidence Scores",
            color_discrete_map={"yolo": "#89b4fa", "retina": "#cba6f7"}
        )
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # ================================================================
    # PAGE 2 — CLUSTER MAP (UMAP)
    # ================================================================
    elif page == "🗺️ Cluster Map":
        st.title("🗺️ Embedding Space — 2D Projection")

        try:
            import umap as umap_lib
            umap_available = True
        except ImportError:
            umap_available = False

        if not umap_available:
            st.warning("UMAP not installed. Run: `pip install umap-learn`")
            st.info("Showing PCA projection instead (less accurate but no install needed)")

            from sklearn.decomposition import PCA
            pca    = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
            method = "PCA"
        else:
            with st.spinner("Computing UMAP projection (first run takes ~30s) ..."):
                X_json = json.dumps(X.tolist())
                coords = compute_umap(X_json)
                method = "UMAP"

        df_plot = df.copy()
        df_plot["x"] = coords[:, 0]
        df_plot["y"] = coords[:, 1]

        view = st.radio(
            "Colour by",
            ["Identity (ground truth)", "Cluster (HDBSCAN)", "Face type"],
            horizontal=True
        )

        if view == "Identity (ground truth)":
            color_col = "identity_label"
            title     = f"{method} — Ground Truth Identities"
        elif view == "Cluster (HDBSCAN)":
            color_col = "cluster_str"
            title     = f"{method} — HDBSCAN Clusters"
        else:
            df_plot["face_type"] = df_plot["is_profile"].map(
                {True: "Profile", False: "Frontal"}
            )
            color_col = "face_type"
            title     = f"{method} — Frontal vs Profile"

        fig = px.scatter(
            df_plot, x="x", y="y",
            color        = color_col,
            hover_data   = ["identity_label", "file_name",
                            "cluster_str", "detector_source",
                            "det_confidence"],
            title        = title,
            template     = "plotly_dark",
            opacity      = 0.75,
            width        = 900, height = 600
        )
        fig.update_traces(marker_size=7)
        fig.update_layout(legend=dict(
            orientation="v", x=1.01, y=1
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Each point = one face embedding. "
            f"Points close together = similar identity in 512D space. "
            f"Well-separated blobs = good clustering."
        )

    # ================================================================
    # PAGE 3 — PER-IDENTITY ANALYSIS
    # ================================================================
    elif page == "👤 Per-Identity Analysis":
        st.title("👤 Per-Identity Cluster Assignment")
        st.markdown("---")

        rows_data = []
        for identity in le.classes_:
            mask       = df["identity_label"] == identity
            assigned   = sorted(df.loc[mask, "cluster"].unique())
            count      = int(mask.sum())
            noise_n    = int((df.loc[mask, "cluster"] == -1).sum())
            clean      = [c for c in assigned if c != -1]
            frontal_n  = int((~df.loc[mask, "is_profile"]).sum())
            profile_n  = int(df.loc[mask, "is_profile"].sum())

            if len(clean) == 1 and noise_n == 0:
                status = "✅ Perfect"
            elif len(clean) == 1:
                status = "~ Partial noise"
            elif len(clean) > 1:
                status = "⚠️ Split"
            else:
                status = "🔕 All noise"

            rows_data.append({
                "Identity"       : identity,
                "Total"          : count,
                "Frontal"        : frontal_n,
                "Profile"        : profile_n,
                "Clusters"       : len(clean),
                "Noise pts"      : noise_n,
                "Assigned"       : str(clean),
                "Status"         : status
            })

        df_id = pd.DataFrame(rows_data)
        st.dataframe(
            df_id.style.applymap(
                lambda v: "color: #a6e3a1" if "Perfect" in str(v)
                else "color: #f9e2af" if "Partial" in str(v)
                else "color: #f38ba8" if "Split" in str(v)
                else "",
                subset=["Status"]
            ),
            use_container_width=True,
            height=400
        )

        # ── Per-identity cluster split chart ──────────────────────
        st.markdown('<div class="section-header">Cluster Count per Identity</div>',
                    unsafe_allow_html=True)
        fig = px.bar(
            df_id, x="Identity", y="Clusters",
            color="Clusters",
            color_continuous_scale="RdYlGn_r",
            title="Number of HDBSCAN clusters per identity (1 = perfect)",
            template="plotly_dark"
        )
        fig.add_hline(
            y=1, line_dash="dash",
            line_color="#a6e3a1",
            annotation_text="Ideal = 1 cluster per identity"
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ── Noise per identity ────────────────────────────────────
        fig2 = px.bar(
            df_id, x="Identity", y="Noise pts",
            color="Noise pts",
            color_continuous_scale="Reds",
            title="Noise points per identity (0 = perfect)",
            template="plotly_dark"
        )
        fig2.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ================================================================
    # PAGE 4 — FRONTAL VS PROFILE
    # ================================================================
    elif page == "📐 Frontal vs Profile":
        st.title("📐 Frontal vs Profile Face Analysis")
        st.markdown("---")

        frontal_mask = ~df["is_profile"].values
        profile_mask =  df["is_profile"].values

        results = {}
        for name, mask in [("Frontal", frontal_mask),
                            ("Profile", profile_mask),
                            ("All",     np.ones(total, dtype=bool))]:
            if mask.sum() < 2:
                continue
            gt  = ground_truths[mask]
            lbl = labels[mask]
            results[name] = {
                "Count"  : int(mask.sum()),
                "NMI"    : round(normalized_mutual_info_score(gt, lbl), 4),
                "ARI"    : round(adjusted_rand_score(gt, lbl), 4),
                "Noise%" : round(float(np.sum(lbl == -1) / mask.sum() * 100), 1)
            }

        # ── Comparison cards ──────────────────────────────────────
        cols = st.columns(3)
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"### {name} Faces")
                st.metric("Count",   res["Count"])
                st.metric("NMI",     res["NMI"])
                st.metric("ARI",     res["ARI"],
                          delta=round(res["ARI"] - results.get(
                              "All", {"ARI": res["ARI"]})["ARI"], 4)
                          if name != "All" else None)
                st.metric("Noise %", f"{res['Noise%']}%")

        st.markdown("---")

        # ── Bar comparison ────────────────────────────────────────
        comp_df = pd.DataFrame([
            {"Category": k, "Metric": "NMI", "Value": v["NMI"]}
            for k, v in results.items()
        ] + [
            {"Category": k, "Metric": "ARI", "Value": v["ARI"]}
            for k, v in results.items()
        ])

        fig = px.bar(
            comp_df, x="Category", y="Value",
            color="Metric", barmode="group",
            title="NMI and ARI — Frontal vs Profile vs All",
            template="plotly_dark",
            color_discrete_map={"NMI": "#89b4fa", "ARI": "#cba6f7"}
        )
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # ── Eye distance distribution ─────────────────────────────
        st.markdown('<div class="section-header">Eye Distance Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(
            df, x="eye_distance",
            color="is_profile",
            nbins=40,
            template="plotly_dark",
            title="Eye Distance (px) — Frontal vs Profile split",
            color_discrete_map={False: "#a6e3a1", True: "#f38ba8"},
            labels={"is_profile": "Is Profile"}
        )
        fig2.add_vline(
            x=20, line_dash="dash",
            line_color="#f9e2af",
            annotation_text="Profile threshold"
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # ================================================================
    # PAGE 5 — RECOMMENDATION SYSTEM
    # ================================================================
    elif page == "🔀 Recommendation System":
        st.title("🔀 Cluster Recommendation System")
        st.markdown("---")

        T1 = st.sidebar.slider("T1 Auto-merge threshold",  0.10, 0.45, 0.30, 0.01)
        T2 = st.sidebar.slider("T2 Prompt threshold",      0.30, 0.70, 0.55, 0.01)

        if not centroid_rows:
            st.warning("No centroids found. Run clustering_v2.py first.")
            return

        # ── Load centroids ────────────────────────────────────────
        centroids = []
        for row in centroid_rows:
            centroids.append({
                "cluster_id"    : row["cluster_id"],
                "identity_label": row["identity_label"],
                "centroid"      : np.array(
                    json.loads(row["centroid_json"]), dtype=np.float32
                ),
                "member_count"  : row["member_count"]
            })

        # ── Centroid distance matrix ──────────────────────────────
        st.markdown('<div class="section-header">Centroid Distance Matrix</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Cosine distance between every pair of cluster centroids. "
            "Low values (green) = clusters likely belong to same identity."
        )

        n_c     = len(centroids)
        labels_c= [f"C{c['cluster_id']}\n{c['identity_label'][:10]}"
                   for c in centroids]
        dist_mat= np.zeros((n_c, n_c))

        for i in range(n_c):
            for j in range(n_c):
                if i != j:
                    dist_mat[i, j] = float(
                        1.0 - np.dot(
                            centroids[i]["centroid"],
                            centroids[j]["centroid"]
                        )
                    )

        fig = px.imshow(
            dist_mat,
            x=labels_c, y=labels_c,
            color_continuous_scale="RdYlGn_r",
            title="Pairwise Cosine Distance Between Cluster Centroids",
            template="plotly_dark",
            zmin=0, zmax=1
        )
        fig.add_shape(
            type="rect", x0=-0.5, y0=-0.5,
            x1=n_c-0.5, y1=n_c-0.5,
            line=dict(color="white", width=0.5)
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # ── Merge candidates ──────────────────────────────────────
        st.markdown('<div class="section-header">Merge Candidates</div>',
                    unsafe_allow_html=True)
        st.caption(
            f"Pairs with cosine distance < {T2} "
            f"(T1 auto-merge < {T1}, T2 prompt < {T2})"
        )

        merge_candidates = []
        for i in range(n_c):
            for j in range(i+1, n_c):
                d = dist_mat[i, j]
                if d < T2:
                    if d < T1:
                        action = "🟢 Auto-merge"
                    else:
                        action = "🟡 Prompt user"
                    merge_candidates.append({
                        "Cluster A"    : f"C{centroids[i]['cluster_id']} — {centroids[i]['identity_label']}",
                        "Cluster B"    : f"C{centroids[j]['cluster_id']} — {centroids[j]['identity_label']}",
                        "Distance"     : round(d, 4),
                        "Members A"    : centroids[i]["member_count"],
                        "Members B"    : centroids[j]["member_count"],
                        "Recommendation": action
                    })

        if merge_candidates:
            mc_df = pd.DataFrame(merge_candidates).sort_values("Distance")
            st.dataframe(mc_df, use_container_width=True, height=350)

            # ── Interactive merge ──────────────────────────────────
            st.markdown('<div class="section-header">Execute Merge</div>',
                        unsafe_allow_html=True)
            st.info(
                "Select two clusters to merge. "
                "This updates the centroid in the database (weighted average)."
            )

            cluster_options = [
                f"C{c['cluster_id']} — {c['identity_label']} ({c['member_count']} members)"
                for c in centroids
            ]

            col1, col2 = st.columns(2)
            with col1:
                sel_a = st.selectbox("Cluster A", cluster_options, key="sel_a")
            with col2:
                sel_b = st.selectbox("Cluster B", cluster_options,
                                     index=min(1, len(cluster_options)-1),
                                     key="sel_b")

            idx_a = cluster_options.index(sel_a)
            idx_b = cluster_options.index(sel_b)

            if idx_a != idx_b:
                ca = centroids[idx_a]
                cb = centroids[idx_b]
                d  = float(1.0 - np.dot(ca["centroid"], cb["centroid"]))
                st.metric("Cosine distance between selected clusters", f"{d:.4f}")

                if d < T1:
                    st.success(f"🟢 T1 — Auto-merge recommended (d < {T1})")
                elif d < T2:
                    st.warning(f"🟡 T2 — User verification recommended")
                else:
                    st.error(f"🔴 T3 — Different identities (d >= {T2})")

                if st.button("✅ Confirm Merge", type="primary"):
                    n_a  = ca["member_count"]
                    n_b  = cb["member_count"]
                    merged = (n_a * ca["centroid"] + n_b * cb["centroid"])
                    norm   = np.linalg.norm(merged)
                    merged = merged / norm if norm > 0 else merged

                    conn   = mysql.connector.connect(**DB_CONFIG)
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE cluster_centroids
                        SET    centroid_json = %s,
                               member_count  = %s,
                               identity_label= %s
                        WHERE  cluster_id = %s
                    """, (
                        json.dumps(merged.tolist()),
                        n_a + n_b,
                        ca["identity_label"],
                        ca["cluster_id"]
                    ))
                    cursor.execute(
                        "DELETE FROM cluster_centroids WHERE cluster_id = %s",
                        (cb["cluster_id"],)
                    )
                    cursor.execute("""
                        INSERT INTO merge_decisions
                            (new_cluster_id, matched_cluster_id,
                             cosine_distance, decision)
                        VALUES (%s, %s, %s, %s)
                    """, (cb["cluster_id"], ca["cluster_id"],
                          d, "user_confirmed"))
                    conn.commit()
                    cursor.close()
                    conn.close()

                    st.success(
                        f"✅ Merged C{cb['cluster_id']} into "
                        f"C{ca['cluster_id']} — "
                        f"new member count: {n_a + n_b}"
                    )
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.success(
                f"✅ No merge candidates found below distance {T2}. "
                f"Clusters are well-separated."
            )

    # ================================================================
    # PAGE 6 — MERGE HISTORY
    # ================================================================
    elif page == "📋 Merge History":
        st.title("📋 Merge Decision History")
        st.markdown("---")

        if not merge_rows:
            st.info("No merge decisions recorded yet.")
            return

        df_merge = pd.DataFrame(merge_rows)
        st.dataframe(df_merge, use_container_width=True)

        if "decision" in df_merge.columns:
            counts = df_merge["decision"].value_counts().reset_index()
            counts.columns = ["Decision", "Count"]
            fig = px.pie(
                counts, names="Decision", values="Count",
                title="Merge Decision Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()