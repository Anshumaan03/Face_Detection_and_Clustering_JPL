import json
import shutil
from pathlib import Path
import mysql.connector

# =====================================================
# CONFIG
# =====================================================

DATASET_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
JSON_FILE = "clustering_results.json"

# ---- MySQL Details ----
MYSQL_HOST = "127.0.0.1"
MYSQL_USER = "root"
MYSQL_PASSWORD = "Anshu@2003"
MYSQL_DATABASE = "face_db"

# =====================================================
# LOAD PRED LABELS
# =====================================================

with open(JSON_FILE, "r") as f:
    clustering_data = json.load(f)

pred_labels = clustering_data["pred_labels"]

print(f"Loaded {len(pred_labels)} cluster labels")

# =====================================================
# LOAD IMAGE LIST FROM MYSQL
# =====================================================

conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE
)

cursor = conn.cursor()

cursor.execute("""
SELECT file_name, identity_label
FROM face_embeddings
ORDER BY id
""")

rows = cursor.fetchall()

cursor.close()
conn.close()

print(f"Loaded {len(rows)} images from MySQL")

# =====================================================
# SAFETY CHECK
# =====================================================

if len(rows) != len(pred_labels):
    raise ValueError(
        f"MySQL rows ({len(rows)}) != "
        f"pred_labels ({len(pred_labels)})"
    )

# =====================================================
# REMOVE OLD RESULTS
# =====================================================

if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)

RESULTS_DIR.mkdir(exist_ok=True)

# =====================================================
# CREATE CLUSTERS
# =====================================================

missing_files = []

for (file_name, identity_label), cluster_id in zip(rows, pred_labels):

    img_path = DATASET_DIR / identity_label / file_name

    if not img_path.exists():
        missing_files.append(str(img_path))
        continue

    if cluster_id == -1:
        cluster_dir = RESULTS_DIR / "Noise"
    else:
        cluster_dir = RESULTS_DIR / f"Cluster_{cluster_id}"

    cluster_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        img_path,
        cluster_dir / file_name
    )

# =====================================================
# REPORT
# =====================================================

print("\n✅ Cluster folders created successfully")
print(f"📁 Output folder: {RESULTS_DIR.resolve()}")

if missing_files:
    print(f"\n⚠ Missing files: {len(missing_files)}")

    for f in missing_files[:20]:
        print("   ", f)