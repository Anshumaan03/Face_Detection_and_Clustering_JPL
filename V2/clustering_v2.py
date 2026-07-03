import json
import numpy as np
import mysql.connector
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder
import hdbscan

# ================================================================
# ⚙️  CONFIGURATION
# ================================================================
DB_CONFIG = {
    "host"    : "127.0.0.1",
    "user"    : "root",
    "password": "Anshu@2003",
    "database": "face_db_v2"
}

# HDBSCAN sweep parameters
MIN_CLUSTER_SIZE_SWEEP = [3, 5, 7, 10]
MIN_SAMPLES_SWEEP      = [1, 2, 3]


# ================================================================
# 🔌  LOAD DATA FROM MYSQL
# ================================================================
def load_embeddings():
    print("🔌 Connecting to MySQL ...")
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT identity_label, embedding_json, is_profile,
               detector_source, det_confidence
        FROM   face_embeddings_v2
        ORDER  BY identity_label, file_name, face_index
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"📥 Loaded {len(rows)} embeddings")
    return rows


# ================================================================
# 🔧  CENTROID UTILITIES
# ================================================================
def spherical_centroid(vectors):
    """
    Correct centroid for L2-normalised vectors on unit hypersphere.
    Element-wise mean then re-normalise.
    """
    c    = np.mean(vectors, axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c


def save_centroids(labels, X, raw_labels, identity_encoder, is_profile_arr):
    """Compute per-cluster spherical centroid and save to DB."""
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE cluster_centroids;")

    unique_clusters = sorted(set(labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)   # skip noise

    for cid in unique_clusters:
        mask    = labels == cid
        vecs    = X[mask]
        centroid= spherical_centroid(vecs)

        # Most common identity in this cluster
        cluster_raw_labels = np.array(raw_labels)[mask]
        unique, counts     = np.unique(cluster_raw_labels, return_counts=True)
        dominant_identity  = unique[np.argmax(counts)]

        frontal_count = int(np.sum(~np.array(is_profile_arr)[mask]))
        profile_count = int(np.sum( np.array(is_profile_arr)[mask]))

        cursor.execute("""
            INSERT INTO cluster_centroids
                (cluster_id, identity_label, centroid_json,
                 member_count, frontal_count, profile_count)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            int(cid),
            dominant_identity,
            json.dumps(centroid.tolist()),
            int(np.sum(mask)),
            frontal_count,
            profile_count
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"  💾 Saved {len(unique_clusters)} cluster centroids to DB")


# ================================================================
# 🚀  MAIN SWEEP
# ================================================================
def run_sweep():
    rows = load_embeddings()
    if not rows:
        print("⚠️  No data found — run pipeline_v2.py first")
        return

    # ── Parse rows ────────────────────────────────────────────────
    raw_labels     = [r[0] for r in rows]
    is_profile_arr = [bool(r[2]) for r in rows]

    X = np.array(
        [json.loads(r[1]) for r in rows],
        dtype=np.float32
    )

    # ── Sanity check — all vectors should be unit length ──────────
    norms = np.linalg.norm(X, axis=1)
    print(f"\n📐 Embedding norm check — mean:{norms.mean():.4f}  "
          f"std:{norms.std():.4f}  (should be ~1.0, ~0.0)")

    # ── Encode labels ─────────────────────────────────────────────
    le             = LabelEncoder()
    ground_truths  = le.fit_transform(raw_labels)
    n_identities   = len(le.classes_)
    total          = len(rows)
    frontal_count  = sum(1 for p in is_profile_arr if not p)
    profile_count  = sum(1 for p in is_profile_arr if p)

    print(f"\n📊 Dataset summary")
    print(f"   Total embeddings : {total}")
    print(f"   Identities       : {n_identities}")
    print(f"   Frontal faces    : {frontal_count} "
          f"({frontal_count/total*100:.1f}%)")
    print(f"   Profile faces    : {profile_count} "
          f"({profile_count/total*100:.1f}%)")
    print(f"   Avg per identity : {total/n_identities:.1f}")

    # ── Silhouette on raw embeddings (no clustering needed) ───────
    print("\n📐 Computing Silhouette Score on raw embeddings ...")
    try:
        sil_raw = silhouette_score(X, ground_truths, metric="cosine")
        print(f"   Silhouette (ground truth labels): {sil_raw:.4f}")
        print("   (measures raw embedding quality — higher = better separated)")
    except Exception as e:
        sil_raw = None
        print(f"   Could not compute: {e}")

    # ── HDBSCAN parametric sweep ──────────────────────────────────
    print(f"\n{'='*95}")
    print("  HDBSCAN PARAMETRIC SWEEP")
    print(f"{'='*95}")
    print(f"  {'min_cls':>7}  {'min_smp':>7}  {'clusters':>9}  "
          f"{'noise':>7}  {'noise%':>7}  {'NMI':>8}  {'ARI':>8}  "
          f"{'Silhouette':>11}")
    print(f"  {'-'*85}")

    best_ari      = -999
    best_cfg      = None
    best_labels   = None

    for min_cls in MIN_CLUSTER_SIZE_SWEEP:
        for min_smp in MIN_SAMPLES_SWEEP:
            if min_smp > min_cls:
                continue

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size     = min_cls,
                min_samples          = min_smp,
                metric               = "euclidean",   # = cosine on L2-norm vecs
                cluster_selection_method = "eom",
                prediction_data      = True
            )
            labels = clusterer.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int(np.sum(labels == -1))
            noise_pct  = n_noise / total * 100

            nmi = normalized_mutual_info_score(ground_truths, labels)
            ari = adjusted_rand_score(ground_truths, labels)

            # Silhouette only meaningful if >= 2 clusters
            if n_clusters >= 2:
                try:
                    # Only on non-noise points
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        sil = silhouette_score(
                            X[non_noise_mask],
                            labels[non_noise_mask],
                            metric="cosine"
                        )
                    else:
                        sil = 0.0
                except Exception:
                    sil = 0.0
            else:
                sil = 0.0

            flag = "  ← BEST" if ari > best_ari else ""
            if ari > best_ari:
                best_ari    = ari
                best_cfg    = (min_cls, min_smp)
                best_labels = labels.copy()

            print(f"  {min_cls:>7}  {min_smp:>7}  {n_clusters:>9}  "
                  f"{n_noise:>7}  {noise_pct:>6.1f}%  "
                  f"{nmi:>8.4f}  {ari:>8.4f}  {sil:>11.4f}{flag}")

    print(f"  {'-'*85}")

    # ── Best config analysis ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🏆 BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"  min_cluster_size = {best_cfg[0]}")
    print(f"  min_samples      = {best_cfg[1]}")
    print()

    n_found  = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise  = int(np.sum(best_labels == -1))
    best_nmi = normalized_mutual_info_score(ground_truths, best_labels)

    print(f"  NMI              : {best_nmi:.4f}")
    print(f"  ARI              : {best_ari:.4f}")
    print(f"  Clusters found   : {n_found}  (expected {n_identities})")
    print(f"  Noise points     : {n_noise} ({n_noise/total*100:.1f}%)")

    # ── Per-identity breakdown ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  📊 PER-IDENTITY CLUSTER ASSIGNMENT")
    print(f"{'='*60}")
    print(f"  {'Identity':<28} {'Imgs':>5}  {'Clusters assigned'}")
    print(f"  {'-'*58}")

    for identity in le.classes_:
        mask     = np.array(raw_labels) == identity
        assigned = sorted(set(best_labels[mask]))
        count    = int(mask.sum())
        noise_n  = int(np.sum(best_labels[mask] == -1))
        clean    = [c for c in assigned if c != -1]

        status = (
            "✅ clean"   if len(clean) == 1 and noise_n == 0 else
            "⚠️  split"   if len(clean) > 1 else
            "🔕 noise"   if len(clean) == 0 else
            "~  partial"
        )
        print(f"  {identity:<28} {count:>5}  {assigned}  {status}")

    # ── Save best centroids to DB ─────────────────────────────────
    print(f"\n💾 Saving cluster centroids to database ...")
    save_centroids(best_labels, X, raw_labels, le, is_profile_arr)

    # ── Profile vs frontal breakdown ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  📐 FRONTAL vs PROFILE FACE ANALYSIS")
    print(f"{'='*60}")

    frontal_mask = ~np.array(is_profile_arr)
    profile_mask =  np.array(is_profile_arr)

    if np.sum(frontal_mask) > 0:
        frontal_nmi = normalized_mutual_info_score(
            ground_truths[frontal_mask],
            best_labels[frontal_mask]
        )
        frontal_ari = adjusted_rand_score(
            ground_truths[frontal_mask],
            best_labels[frontal_mask]
        )
        print(f"  Frontal faces NMI : {frontal_nmi:.4f}")
        print(f"  Frontal faces ARI : {frontal_ari:.4f}")

    if np.sum(profile_mask) > 0:
        profile_nmi = normalized_mutual_info_score(
            ground_truths[profile_mask],
            best_labels[profile_mask]
        )
        profile_ari = adjusted_rand_score(
            ground_truths[profile_mask],
            best_labels[profile_mask]
        )
        print(f"  Profile faces NMI : {profile_nmi:.4f}")
        print(f"  Profile faces ARI : {profile_ari:.4f}")

    print()
    print(f"  ✅ Done — run recommendation_system.py for interactive merging")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_sweep()