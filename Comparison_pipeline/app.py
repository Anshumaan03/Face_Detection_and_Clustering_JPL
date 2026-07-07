import os
import glob

import numpy as np
import streamlit as st
import pandas as pd

import config
from storage import Storage
from clustering import cluster_all_models, metrics_summary_table
from common import detect_faces, largest_or_best_face, align_face
from embeddings import EXTRACTOR_CLASSES, get_normalized_embedding
from recommendation import (
    recommend_for_embedding,
    assign_face_to_cluster,
    create_new_cluster,
    reclaim_noise,
    pairwise_cluster_scan,
    merge_clusters,
    calibrate_thresholds,
)

st.set_page_config(page_title="Face Embedding Comparison", layout="wide")
st.title("Face Embedding Model Comparison")
st.caption("ArcFace · dlib-ResNet · FaceNet (VGGFace2) · SigLIP2 — common detector, per-model alignment, identical HDBSCAN")

run_label = st.sidebar.text_input("Run label", value="default")

if st.sidebar.button("Re-run clustering (uses existing DB embeddings)"):
    with st.spinner("Running HDBSCAN for all models..."):
        results = cluster_all_models(run_label=run_label)
        st.session_state["results"] = results
    st.sidebar.warning(
        "Re-clustering just rebuilt cluster_results + centroids from scratch. "
        "Any manual merges / noise reclamation you'd done for this run_label are now gone -- "
        "that's expected, HDBSCAN has no memory of them."
    )

if "results" not in st.session_state:
    with st.spinner("Loading clustering results..."):
        st.session_state["results"] = cluster_all_models(run_label=run_label)

results = st.session_state["results"]


@st.cache_resource(show_spinner=False)
def get_extractor(model_name: str):
    return EXTRACTOR_CLASSES[model_name]()


tab_metrics, tab_flow1, tab_flow2, tab_noise, tab_calibrate = st.tabs([
    "Metrics & clusters",
    "Recommend (Flow 1)",
    "Pairwise check (Flow 2)",
    "Noise reclamation",
    "Calibrate thresholds",
])

# ===========================================================================
# TAB: Metrics + cluster browser (original app content)
# ===========================================================================
with tab_metrics:
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

# ===========================================================================
# TAB: Flow 1 -- upload a new face, compare to all centroids
# ===========================================================================
with tab_flow1:
    st.subheader("Upload a new face and get a cluster recommendation")
    model_choice_1 = st.selectbox("Model", list(config.MODEL_INPUT_SPECS.keys()), key="flow1_model")
    uploaded_file = st.file_uploader("Upload an image containing one face", type=["jpg", "jpeg", "png"], key="flow1_upload")

    if uploaded_file is not None and st.button("Analyze", key="flow1_analyze"):
        import cv2

        raw_bytes = uploaded_file.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

        faces = detect_faces(img_bgr)
        face = largest_or_best_face(faces)

        if face is None:
            st.error("No face detected in the uploaded image.")
            st.session_state.pop("flow1_pending", None)
        else:
            aligned = align_face(img_bgr, face, model=model_choice_1)
            extractor = get_extractor(model_choice_1)
            vec = get_normalized_embedding(extractor, aligned)

            db = Storage()
            rec = recommend_for_embedding(vec, model_choice_1, run_label, db)
            db.close()

            os.makedirs(config.UPLOADS_ROOT, exist_ok=True)
            save_path = os.path.join(config.UPLOADS_ROOT, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(raw_bytes)

            st.session_state["flow1_pending"] = {
                "rec": rec,
                "vec": vec,
                "save_path": save_path,
                "model": model_choice_1,
            }

    if "flow1_pending" in st.session_state:
        pending = st.session_state["flow1_pending"]
        rec = pending["rec"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(pending["save_path"], caption="Uploaded image", use_container_width=True)
        with col_b:
            if rec["representative_image_path"]:
                st.image(rec["representative_image_path"],
                          caption=f"Nearest cluster {rec['nearest_cluster_label']} "
                                  f"(identity: {rec['representative_identity']})",
                          use_container_width=True)
            else:
                st.info("No existing clusters yet for this model/run -- this will start the first one.")

        if rec["distance"] is not None:
            st.write(f"Cosine distance to nearest centroid: **{rec['distance']:.3f}**")

        if rec["status"] == "auto_merge":
            st.success(f"High confidence match -- auto-merge into cluster {rec['nearest_cluster_label']}.")
            if st.button("Confirm auto-merge", key="flow1_confirm_auto"):
                db = Storage()
                face_id = db.insert_face(
                    identity="uploaded", image_path=pending["save_path"],
                    bbox=np.zeros(4), landmarks=np.zeros((5, 2)), det_score=1.0,
                )
                db.insert_embedding(face_id, pending["model"], pending["vec"])
                assign_face_to_cluster(face_id, pending["model"], run_label,
                                        rec["nearest_cluster_label"], pending["vec"], db)
                db.close()
                st.success(f"Assigned to cluster {rec['nearest_cluster_label']}.")
                del st.session_state["flow1_pending"]

        elif rec["status"] == "ask_user":
            st.warning("Borderline match -- is this the same person as the cluster shown above?")
            c1, c2 = st.columns(2)
            if c1.button("Yes, same person", key="flow1_yes"):
                db = Storage()
                face_id = db.insert_face(
                    identity="uploaded", image_path=pending["save_path"],
                    bbox=np.zeros(4), landmarks=np.zeros((5, 2)), det_score=1.0,
                )
                db.insert_embedding(face_id, pending["model"], pending["vec"])
                assign_face_to_cluster(face_id, pending["model"], run_label,
                                        rec["nearest_cluster_label"], pending["vec"], db)
                db.close()
                st.success(f"Assigned to cluster {rec['nearest_cluster_label']}.")
                del st.session_state["flow1_pending"]
            if c2.button("No, different person", key="flow1_no"):
                db = Storage()
                face_id = db.insert_face(
                    identity="uploaded", image_path=pending["save_path"],
                    bbox=np.zeros(4), landmarks=np.zeros((5, 2)), det_score=1.0,
                )
                db.insert_embedding(face_id, pending["model"], pending["vec"])
                new_label = create_new_cluster(face_id, pending["model"], run_label, pending["vec"], db)
                db.close()
                st.success(f"Created new cluster {new_label}.")
                del st.session_state["flow1_pending"]

        else:  # new_cluster
            st.info("No close match found -- this will start a new cluster.")
            if st.button("Confirm new cluster", key="flow1_confirm_new"):
                db = Storage()
                face_id = db.insert_face(
                    identity="uploaded", image_path=pending["save_path"],
                    bbox=np.zeros(4), landmarks=np.zeros((5, 2)), det_score=1.0,
                )
                db.insert_embedding(face_id, pending["model"], pending["vec"])
                new_label = create_new_cluster(face_id, pending["model"], run_label, pending["vec"], db)
                db.close()
                st.success(f"Created new cluster {new_label}.")
                del st.session_state["flow1_pending"]

# ===========================================================================
# TAB: Flow 2 -- pairwise cluster-vs-cluster sweep
# ===========================================================================
with tab_flow2:
    st.subheader("Scan existing clusters for suspicious pairs (possible same person, split across clusters)")
    model_choice_2 = st.selectbox("Model", list(config.MODEL_INPUT_SPECS.keys()), key="flow2_model")

    if st.button("Scan for suspicious cluster pairs", key="flow2_scan"):
        db = Storage()
        st.session_state["flow2_suspicious"] = pairwise_cluster_scan(model_choice_2, run_label, db)
        st.session_state["flow2_model_used"] = model_choice_2
        db.close()

    if "flow2_suspicious" in st.session_state:
        suspicious = st.session_state["flow2_suspicious"]
        used_model = st.session_state.get("flow2_model_used", model_choice_2)

        if not suspicious:
            st.success("No suspicious cluster pairs found within the threshold zone.")
        else:
            st.write(f"Found **{len(suspicious)}** suspicious pair(s) for model `{used_model}`.")
            for pair in suspicious:
                key = f"{pair['cluster_a']}_{pair['cluster_b']}"
                st.markdown("---")
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    st.image(pair["rep_a_image_path"], caption=f"Cluster {pair['cluster_a']}", use_container_width=True)
                with c2:
                    st.image(pair["rep_b_image_path"], caption=f"Cluster {pair['cluster_b']}", use_container_width=True)
                with c3:
                    st.write(f"Distance: **{pair['distance']:.3f}**"
                             + (" (auto-mergeable)" if pair["auto_mergeable"] else ""))
                    b1, b2 = st.columns(2)
                    if b1.button("Merge (same person)", key=f"flow2_merge_{key}"):
                        db = Storage()
                        merge_result = merge_clusters(used_model, run_label, pair["cluster_a"], pair["cluster_b"], db)
                        db.close()
                        st.success(f"Merged cluster {merge_result['merged_away']} into "
                                   f"{merge_result['kept']} ({merge_result['n_members']} members).")
                        st.session_state["flow2_suspicious"] = [
                            s for s in suspicious
                            if not (s["cluster_a"] == pair["cluster_a"] and s["cluster_b"] == pair["cluster_b"])
                        ]
                        st.rerun()
                    if b2.button("Not the same person", key=f"flow2_reject_{key}"):
                        st.session_state["flow2_suspicious"] = [
                            s for s in suspicious
                            if not (s["cluster_a"] == pair["cluster_a"] and s["cluster_b"] == pair["cluster_b"])
                        ]
                        st.rerun()

# ===========================================================================
# TAB: Noise reclamation
# ===========================================================================
with tab_noise:
    st.subheader("Reclaim noise points (HDBSCAN's -1 bucket)")
    st.caption("Noise points are real faces HDBSCAN couldn't confidently place in any dense cluster -- "
               "not garbage. This routes each one through the same Flow-1 logic against real clusters.")
    model_choice_3 = st.selectbox("Model", list(config.MODEL_INPUT_SPECS.keys()), key="noise_model")

    if st.button("Scan noise points", key="noise_scan"):
        db = Storage()
        st.session_state["noise_result"] = reclaim_noise(model_choice_3, run_label, db, auto_apply=True)
        st.session_state["noise_model_used"] = model_choice_3
        db.close()

    if "noise_result" in st.session_state:
        result = st.session_state["noise_result"]
        used_model = st.session_state.get("noise_model_used", model_choice_3)
        st.write(f"Model `{used_model}`: **{len(result['auto_merged'])}** auto-merged, "
                 f"**{len(result['suggestions'])}** need your review, "
                 f"**{len(result['left_as_noise'])}** left as noise.")

        remaining_suggestions = []
        for s in result["suggestions"]:
            st.markdown("---")
            db = Storage()
            face_info = db.get_face_info(s["face_id"])
            db.close()

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.image(face_info["image_path"], caption="Noise point", use_container_width=True)
            with c2:
                st.image(s["representative_image_path"], caption=f"Cluster {s['nearest_cluster_label']}", use_container_width=True)
            with c3:
                st.write(f"Distance: **{s['distance']:.3f}**")
                b1, b2 = st.columns(2)
                resolved = False
                if b1.button("Yes, same person", key=f"noise_yes_{s['face_id']}"):
                    db = Storage()
                    vec = db.get_embedding_vector(s["face_id"], used_model)
                    assign_face_to_cluster(s["face_id"], used_model, run_label, s["nearest_cluster_label"], vec, db)
                    db.close()
                    st.success("Assigned to cluster.")
                    resolved = True
                if b2.button("No, leave as noise", key=f"noise_no_{s['face_id']}"):
                    st.info("Left as noise.")
                    resolved = True
                if not resolved:
                    remaining_suggestions.append(s)

        result["suggestions"] = remaining_suggestions
        st.session_state["noise_result"] = result

# ===========================================================================
# TAB: Threshold calibration
# ===========================================================================
with tab_calibrate:
    st.subheader("Data-driven threshold calibration")
    st.caption(
        "config.THRESHOLDS starts with generic guesses per model. This uses the ground-truth "
        "identity labels already in your dataset (from the data/raw/<identity>/ folders) to suggest "
        "better T1 (same-identity, 95th percentile) / T2 (different-identity, 5th percentile) values -- "
        "an offline tuning step you wouldn't have without labelled data."
    )

    if st.button("Calibrate all models", key="calibrate_run"):
        db = Storage()
        rows = []
        for m in config.MODEL_INPUT_SPECS:
            try:
                rows.append(calibrate_thresholds(m, run_label, db))
            except ValueError as e:
                st.warning(f"{m}: {e}")
        db.close()
        if rows:
            st.session_state["calibration_rows"] = rows

    if "calibration_rows" in st.session_state:
        cal_df = pd.DataFrame(st.session_state["calibration_rows"]).set_index("model")
        st.dataframe(cal_df.style.format(
            "{:.3f}",
            subset=["same_identity_dist_mean", "diff_identity_dist_mean", "suggested_t1", "suggested_t2"],
        ))
        for _, r in cal_df.reset_index().iterrows():
            if not r["clean_separation"]:
                st.warning(f"`{r['model']}`: suggested T1 ({r['suggested_t1']:.3f}) > suggested T2 "
                           f"({r['suggested_t2']:.3f}) -- no clean single global cutoff for this model on this data.")
        st.info("Copy the suggested_t1 / suggested_t2 values you're happy with into config.THRESHOLDS.")