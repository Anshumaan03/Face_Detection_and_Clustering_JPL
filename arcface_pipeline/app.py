import os
import glob

import numpy as np
import streamlit as st
import pandas as pd

import config
from storage import Storage
from clustering import cluster
from common import detect_faces, largest_or_best_face, align_face
from embedding import ArcFaceExtractor, get_normalized_embedding
from recommendation import (
    recommend_for_embedding,
    assign_face_to_cluster,
    create_new_cluster,
    reclaim_noise,
    pairwise_cluster_scan,
    merge_clusters,
    calibrate_thresholds,
)

st.set_page_config(page_title="Face Clustering & Recommendation", layout="wide")
st.title("Face Clustering & Recommendation System")
st.caption("Detector: InsightFace (buffalo_l)  ·  Embedding: ArcFace (512-d)  ·  Clustering: HDBSCAN")

run_label = st.sidebar.text_input("Run label", value="default")

if st.sidebar.button("Re-run clustering (uses existing DB embeddings)"):
    with st.spinner("Running HDBSCAN..."):
        st.session_state["cluster_result"] = cluster(run_label=run_label)
    st.sidebar.warning(
        "Re-clustering just rebuilt cluster_results + centroids from scratch. "
        "Any manual merges / noise reclamation you'd done for this run_label are now gone -- "
        "that's expected, HDBSCAN has no memory of them."
    )

if "cluster_result" not in st.session_state:
    with st.spinner("Loading clustering results..."):
        try:
            st.session_state["cluster_result"] = cluster(run_label=run_label)
        except ValueError as e:
            st.session_state["cluster_result"] = None
            st.warning(f"{e} Run the extraction pipeline first (runpipeline.py), then reload.")

cluster_result = st.session_state["cluster_result"]


@st.cache_resource(show_spinner=False)
def get_extractor():
    return ArcFaceExtractor()


def _insert_uploaded_face(save_path: str, vec: np.ndarray):
    db = Storage()
    face_id = db.insert_face(
        identity="uploaded", image_path=save_path,
        bbox=np.zeros(4), landmarks=np.zeros((5, 2)), det_score=1.0,
    )
    db.insert_embedding(face_id, vec)
    return db, face_id


tab_metrics, tab_flow1, tab_flow2, tab_noise, tab_calibrate = st.tabs([
    "Metrics & clusters",
    "Recommend (Flow 1)",
    "Pairwise check (Flow 2)",
    "Noise reclamation",
    "Calibrate thresholds",
])

# ===========================================================================
# TAB: Metrics + cluster browser
# ===========================================================================
with tab_metrics:
    if cluster_result is None:
        st.info("No embeddings in the database yet.")
    else:
        st.subheader("Clustering metrics")
        metrics = cluster_result["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ARI", f"{metrics['ari']:.3f}")
        m2.metric("NMI", f"{metrics['nmi']:.3f}")
        m3.metric("Clusters found", metrics["n_predicted_clusters"])
        m4.metric("Noise fraction", f"{metrics['noise_fraction']:.1%}")
        with st.expander("Full metrics"):
            st.json(metrics)

        st.subheader("Cluster browser")
        cluster_dir = os.path.join(config.CLUSTERS_ROOT, run_label)

        if not os.path.isdir(cluster_dir):
            st.warning(f"No cluster folder found at {cluster_dir}. Run the pipeline / clustering first.")
        else:
            cluster_folders = sorted(
                os.listdir(cluster_dir),
                key=lambda x: (x != "noise", x)
            )
            cluster_choice = st.selectbox("Cluster", cluster_folders, key="browse_cluster")

            chosen_dir = os.path.join(cluster_dir, cluster_choice)
            image_paths = sorted(glob.glob(os.path.join(chosen_dir, "*")))

            st.write(f"**{len(image_paths)} images** in `{cluster_choice}`")

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
    uploaded_file = st.file_uploader("Upload an image containing one face", type=["jpg", "jpeg", "png"], key="flow1_upload")

    if uploaded_file is not None and st.button("Analyze", key="flow1_analyze"):
        import cv2

        raw_bytes = uploaded_file.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

        faces = detect_faces(img_bgr)
        face = largest_or_best_face(faces)

        st.session_state.pop("flow1_pending", None)
        st.session_state.pop("flow1_result_message", None)

        if face is None:
            st.session_state.pop("flow1_last_analysis", None)
            st.error("No face detected in the uploaded image.")
        else:
            aligned = align_face(img_bgr, face)
            extractor = get_extractor()
            vec = get_normalized_embedding(extractor, aligned)

            # Diagnostic: which face (of possibly several) got used, and what did the
            # actual aligned crop look like -- this is the fastest way to catch a wrong
            # face being picked out of a multi-face photo (see representative image note
            # below, e.g. two people in frame, or a photo-within-a-photo).
            st.session_state["flow1_last_analysis"] = {
                "n_faces": len(faces),
                "det_score": face.det_score,
                "aligned_rgb": cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
            }

            db = Storage()
            rec = recommend_for_embedding(vec, run_label, db)
            db.close()

            os.makedirs(config.UPLOADS_ROOT, exist_ok=True)
            save_path = os.path.join(config.UPLOADS_ROOT, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(raw_bytes)

            # -------------------------------------------------------------
            # Strict three-way rule: only the borderline (ask_user) zone
            # waits for a human decision. auto_merge and new_cluster both
            # happen immediately -- that's the whole point of calibrating
            # T1/T2 in the first place.
            # -------------------------------------------------------------
            if rec["status"] == "auto_merge":
                db2, face_id = _insert_uploaded_face(save_path, vec)
                assign_face_to_cluster(face_id, run_label, rec["nearest_cluster_label"], vec, db2)
                db2.close()
                st.session_state["flow1_result_message"] = (
                    "success",
                    f"This image is clustered into cluster {rec['nearest_cluster_label']} "
                    f"(identity: {rec['representative_identity']}) -- auto-merged "
                    f"(distance {rec['distance']:.3f})."
                )
            elif rec["status"] == "new_cluster":
                db2, face_id = _insert_uploaded_face(save_path, vec)
                new_label = create_new_cluster(face_id, run_label, vec, db2)
                db2.close()
                dist_note = f" (nearest existing cluster was {rec['distance']:.3f} away)" if rec["distance"] is not None else ""
                st.session_state["flow1_result_message"] = (
                    "info",
                    f"This image did not match any existing cluster{dist_note} -- "
                    f"new cluster {new_label} has been formed."
                )
            else:  # ask_user
                st.session_state["flow1_pending"] = {"rec": rec, "vec": vec, "save_path": save_path}

    # Diagnostic panel: always visible after an analysis, regardless of which
    # branch it took. If the wrong face got picked out of a multi-face photo,
    # this is where you'd see it.
    if "flow1_last_analysis" in st.session_state:
        diag = st.session_state["flow1_last_analysis"]
        if diag["n_faces"] > 1:
            st.warning(f"{diag['n_faces']} faces were detected in the uploaded image -- the "
                       f"highest-confidence one (det_score={diag['det_score']:.3f}) was used. "
                       f"If that's not the person you meant, crop the image down to just them "
                       f"before uploading.")
        with st.expander("Show the exact aligned crop that was embedded"):
            st.image(diag["aligned_rgb"], width=150, caption="112x112 crop fed into ArcFace")

    if "flow1_result_message" in st.session_state:
        kind, msg = st.session_state["flow1_result_message"]
        getattr(st, kind)(msg)

    if "flow1_pending" in st.session_state:
        pending = st.session_state["flow1_pending"]
        rec = pending["rec"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(pending["save_path"], caption="Uploaded image", use_container_width=True)
        with col_b:
            st.image(rec["representative_image_path"],
                      caption=f"Nearest cluster {rec['nearest_cluster_label']} "
                              f"(identity: {rec['representative_identity']})",
                      use_container_width=True)

        st.warning(f"This image is asking for confirmation -- cosine distance to cluster "
                   f"{rec['nearest_cluster_label']} is **{rec['distance']:.3f}**. Same person?")

        c1, c2 = st.columns(2)
        if c1.button("Yes, same person", key="flow1_yes"):
            db, face_id = _insert_uploaded_face(pending["save_path"], pending["vec"])
            assign_face_to_cluster(face_id, run_label, rec["nearest_cluster_label"], pending["vec"], db)
            db.close()
            st.session_state["flow1_result_message"] = (
                "success",
                f"This image is clustered into cluster {rec['nearest_cluster_label']} "
                f"(identity: {rec['representative_identity']})."
            )
            del st.session_state["flow1_pending"]
            st.rerun()
        if c2.button("No, different person", key="flow1_no"):
            db, face_id = _insert_uploaded_face(pending["save_path"], pending["vec"])
            new_label = create_new_cluster(face_id, run_label, pending["vec"], db)
            db.close()
            st.session_state["flow1_result_message"] = (
                "info",
                f"This image did not match -- new cluster {new_label} has been formed."
            )
            del st.session_state["flow1_pending"]
            st.rerun()

# ===========================================================================
# TAB: Flow 2 -- pairwise cluster-vs-cluster sweep
# ===========================================================================
with tab_flow2:
    st.subheader("Scan existing clusters for suspicious pairs (possible same person, split across clusters)")

    if st.button("Scan for suspicious cluster pairs", key="flow2_scan"):
        db = Storage()
        st.session_state["flow2_suspicious"] = pairwise_cluster_scan(run_label, db)
        db.close()

    if "flow2_suspicious" in st.session_state:
        suspicious = st.session_state["flow2_suspicious"]

        if not suspicious:
            st.success("No suspicious cluster pairs found within the threshold zone.")
        else:
            st.write(f"Found **{len(suspicious)}** suspicious pair(s).")
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
                        merge_result = merge_clusters(run_label, pair["cluster_a"], pair["cluster_b"], db)
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

    if st.button("Scan noise points", key="noise_scan"):
        db = Storage()
        st.session_state["noise_result"] = reclaim_noise(run_label, db, auto_apply=True)
        db.close()

    if "noise_result" in st.session_state:
        result = st.session_state["noise_result"]
        st.write(f"**{len(result['auto_merged'])}** auto-merged, "
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
                    vec = db.get_embedding_vector(s["face_id"])
                    assign_face_to_cluster(s["face_id"], run_label, s["nearest_cluster_label"], vec, db)
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
        "config.THRESHOLDS starts with a generic guess. This uses the ground-truth identity "
        "labels already in your dataset (from the data/raw/<identity>/ folders) to suggest better "
        "T1 (same-identity, 95th percentile) / T2 (different-identity, 5th percentile) values -- "
        "an offline tuning step you wouldn't have without labelled data."
    )

    if st.button("Calibrate", key="calibrate_run"):
        db = Storage()
        try:
            st.session_state["calibration_result"] = calibrate_thresholds(run_label, db)
        except ValueError as e:
            st.warning(str(e))
        db.close()

    if "calibration_result" in st.session_state:
        r = st.session_state["calibration_result"]
        c1, c2 = st.columns(2)
        c1.metric("Suggested T1 (auto-merge below)", f"{r['suggested_t1']:.3f}")
        c2.metric("Suggested T2 (new cluster above)", f"{r['suggested_t2']:.3f}")
        st.write(f"Same-identity mean distance: {r['same_identity_dist_mean']:.3f}  |  "
                 f"Different-identity mean distance: {r['diff_identity_dist_mean']:.3f}  |  "
                 f"Sampled {r['n_same_pairs']} same-id and {r['n_diff_pairs']} diff-id pairs.")
        if not r["clean_separation"]:
            st.warning(f"Suggested T1 ({r['suggested_t1']:.3f}) > suggested T2 ({r['suggested_t2']:.3f}) -- "
                       f"no clean single global cutoff for ArcFace on this data.")
        st.info("Copy suggested_t1 / suggested_t2 into config.THRESHOLDS once you're happy with them.")