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

    def close(self):
        self.conn.close()
