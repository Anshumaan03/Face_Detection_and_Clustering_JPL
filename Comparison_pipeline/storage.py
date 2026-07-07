"""
Schema
------
faces            : one row per (image, detected face) — bbox, landmarks, det_score
embeddings       : one row per (face, model) — the actual vector + dim, as JSON
                    (JSON keeps it human-inspectable; fine at this dataset scale)

"""

from __future__ import annotations

import json
import logging
from typing import Optional, List

import numpy as np
import mysql.connector
from mysql.connector import errorcode

import config

logger = logging.getLogger(__name__)


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS faces (
        id INT AUTO_INCREMENT PRIMARY KEY,
        identity VARCHAR(255) NOT NULL,
        image_path VARCHAR(1024) NOT NULL,
        bbox JSON NOT NULL,
        landmarks JSON NOT NULL,
        det_score FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_image_path (image_path(768))
    ) ENGINE=InnoDB;
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        face_id INT NOT NULL,
        model VARCHAR(64) NOT NULL,
        dim INT NOT NULL,
        vector JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
        UNIQUE KEY uq_face_model (face_id, model)
    ) ENGINE=InnoDB;
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster_results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(64) NOT NULL,
        run_label VARCHAR(255) NOT NULL,
        face_id INT NOT NULL,
        cluster_label INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
        UNIQUE KEY uq_run_face (model, run_label, face_id)
    ) ENGINE=InnoDB;
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster_centroids (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(64) NOT NULL,
        run_label VARCHAR(255) NOT NULL,
        cluster_label INT NOT NULL,
        centroid_vector JSON NOT NULL,
        n_members INT NOT NULL,
        representative_face_id INT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (representative_face_id) REFERENCES faces(id) ON DELETE CASCADE,
        UNIQUE KEY uq_model_run_cluster (model, run_label, cluster_label)
    ) ENGINE=InnoDB;
    """,
]


class Storage:
    def __init__(self, db_config: dict = None):
        self.db_config = db_config or config.MYSQL_CONFIG
        self._ensure_database_exists()
        self.conn = mysql.connector.connect(**self.db_config)

    def _ensure_database_exists(self):
        """Creates the target database if it doesn't exist yet (connects without `database` first)."""
        cfg_no_db = {k: v for k, v in self.db_config.items() if k != "database"}
        conn = mysql.connector.connect(**cfg_no_db)
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.db_config['database']}` DEFAULT CHARACTER SET utf8mb4;")
        conn.commit()
        cur.close()
        conn.close()

    def create_schema(self):
        cur = self.conn.cursor()
        for stmt in SCHEMA_STATEMENTS:
            cur.execute(stmt)
        self.conn.commit()
        cur.close()
        logger.info("Schema ensured (faces, embeddings, cluster_results).")

    # -- faces ---------------------------------------------------------

    def insert_face(self, identity: str, image_path: str, bbox: np.ndarray,
                     landmarks: np.ndarray, det_score: float) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO faces (identity, image_path, bbox, landmarks, det_score)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                bbox = VALUES(bbox), landmarks = VALUES(landmarks), det_score = VALUES(det_score)
            """,
            (identity, image_path, json.dumps(np.asarray(bbox).tolist()),
             json.dumps(np.asarray(landmarks).tolist()), float(det_score)),
        )
        self.conn.commit()
        if cur.lastrowid:
            face_id = cur.lastrowid
        else:
            cur.execute("SELECT id FROM faces WHERE image_path = %s", (image_path,))
            face_id = cur.fetchone()[0]
        cur.close()
        return face_id

    # -- embeddings ------------------------------------------------------

    def insert_embedding(self, face_id: int, model: str, vector: np.ndarray):
        cur = self.conn.cursor()
        vec_list = np.asarray(vector, dtype=np.float32).tolist()
        cur.execute(
            """
            INSERT INTO embeddings (face_id, model, dim, vector)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE vector = VALUES(vector), dim = VALUES(dim)
            """,
            (face_id, model, len(vec_list), json.dumps(vec_list)),
        )
        self.conn.commit()
        cur.close()

    def load_embeddings_df(self, model: str):
        """Returns a pandas DataFrame: face_id, identity, image_path, vector (np.ndarray)."""
        import pandas as pd
        cur = self.conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT f.id AS face_id, f.identity, f.image_path, e.vector
            FROM embeddings e JOIN faces f ON f.id = e.face_id
            WHERE e.model = %s
            ORDER BY f.id
            """,
            (model,),
        )
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r["vector"] = np.array(json.loads(r["vector"]), dtype=np.float32)
        return pd.DataFrame(rows)

    # -- cluster results ---------------------------------------------------

    def insert_cluster_labels(self, model: str, run_label: str, face_ids: List[int], labels: List[int]):
        cur = self.conn.cursor()
        rows = [(model, run_label, fid, int(lbl)) for fid, lbl in zip(face_ids, labels)]
        cur.executemany(
            """
            INSERT INTO cluster_results (model, run_label, face_id, cluster_label)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE cluster_label = VALUES(cluster_label)
            """,
            rows,
        )
        self.conn.commit()
        cur.close()

    # -- misc lookups needed by recommendation.py -------------------------

    def get_embedding_vector(self, face_id: int, model: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT vector FROM embeddings WHERE face_id = %s AND model = %s", (face_id, model))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return np.array(json.loads(row["vector"]), dtype=np.float32)

    def get_face_info(self, face_id: int) -> Optional[dict]:
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT id, identity, image_path FROM faces WHERE id = %s", (face_id,))
        row = cur.fetchone()
        cur.close()
        return row

    def get_cluster_face_ids(self, model: str, run_label: str, cluster_label: int) -> List[int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT face_id FROM cluster_results WHERE model = %s AND run_label = %s AND cluster_label = %s",
            (model, run_label, cluster_label),
        )
        ids = [r[0] for r in cur.fetchall()]
        cur.close()
        return ids

    def get_noise_face_ids(self, model: str, run_label: str) -> List[int]:
        return self.get_cluster_face_ids(model, run_label, -1)

    def get_all_cluster_labels(self, model: str, run_label: str) -> List[int]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT cluster_label FROM cluster_results
            WHERE model = %s AND run_label = %s AND cluster_label != -1
            """,
            (model, run_label),
        )
        labels = sorted(r[0] for r in cur.fetchall())
        cur.close()
        return labels

    def set_face_cluster_label(self, model: str, run_label: str, face_id: int, cluster_label: int):
        """Writes/overwrites a single face's cluster assignment (used by Flow 1 + noise reclamation)."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO cluster_results (model, run_label, face_id, cluster_label)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE cluster_label = VALUES(cluster_label)
            """,
            (model, run_label, face_id, cluster_label),
        )
        self.conn.commit()
        cur.close()

    def reassign_cluster_label(self, model: str, run_label: str, old_label: int, new_label: int):
        """Bulk-moves every face at old_label to new_label (used when merging two clusters)."""
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE cluster_results SET cluster_label = %s
            WHERE model = %s AND run_label = %s AND cluster_label = %s
            """,
            (new_label, model, run_label, old_label),
        )
        self.conn.commit()
        cur.close()

    # -- cluster centroids (Flow 1 + Flow 2 rely on these) -----------------

    def upsert_centroid(self, model: str, run_label: str, cluster_label: int,
                         centroid_vector: np.ndarray, n_members: int, representative_face_id: int):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO cluster_centroids
                (model, run_label, cluster_label, centroid_vector, n_members, representative_face_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                centroid_vector = VALUES(centroid_vector),
                n_members = VALUES(n_members),
                representative_face_id = VALUES(representative_face_id)
            """,
            (model, run_label, cluster_label,
             json.dumps(np.asarray(centroid_vector, dtype=np.float32).tolist()),
             int(n_members), int(representative_face_id)),
        )
        self.conn.commit()
        cur.close()

    def delete_centroid(self, model: str, run_label: str, cluster_label: int):
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM cluster_centroids WHERE model = %s AND run_label = %s AND cluster_label = %s",
            (model, run_label, cluster_label),
        )
        self.conn.commit()
        cur.close()

    def delete_all_centroids(self, model: str, run_label: str):
        """Wipes centroids for a (model, run_label) before rebuilding from a fresh HDBSCAN run."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM cluster_centroids WHERE model = %s AND run_label = %s", (model, run_label))
        self.conn.commit()
        cur.close()

    def load_centroids_df(self, model: str, run_label: str):
        """Returns a DataFrame: cluster_label, centroid_vector (np.ndarray), n_members,
        representative_face_id, representative_identity, representative_image_path."""
        import pandas as pd
        cur = self.conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT cc.cluster_label, cc.centroid_vector, cc.n_members, cc.representative_face_id,
                   f.identity AS representative_identity, f.image_path AS representative_image_path
            FROM cluster_centroids cc
            JOIN faces f ON f.id = cc.representative_face_id
            WHERE cc.model = %s AND cc.run_label = %s
            ORDER BY cc.cluster_label
            """,
            (model, run_label),
        )
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r["centroid_vector"] = np.array(json.loads(r["centroid_vector"]), dtype=np.float32)
        return pd.DataFrame(rows)

    def close(self):
        self.conn.close()