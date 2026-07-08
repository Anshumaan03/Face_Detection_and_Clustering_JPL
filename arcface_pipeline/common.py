"""
common.py
=========
Shared image loading + face detection (InsightFace RetinaFace, buffalo_l),
and ArcFace alignment (5-point similarity-transform warp to the canonical
112x112 template). This pipeline is now single-model (ArcFace only), so
there is exactly one alignment branch — no per-model dispatch needed.

Typical usage
-------------
    from common import load_image, detect_faces, largest_or_best_face, align_face

    img = load_image(path)
    faces = detect_faces(img)
    face = largest_or_best_face(faces)
    aligned_112 = align_face(img, face)   # ready for embeddings.ArcFaceExtractor
"""

from __future__ import annotations

import os
import glob
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coords, original image
    landmarks: np.ndarray     # shape (5, 2): left_eye, right_eye, nose, mouth_l, mouth_r
    det_score: float
    image_path: str = ""


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Loads an image as BGR uint8 (OpenCV convention, matches InsightFace's expectation)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read image (corrupt or unsupported): {path}")
    return img


def iter_dataset(data_root: str = config.DATA_ROOT):
    """
    Yields (identity_label, image_path) for every image under
    data_root/<identity>/*.{jpg,jpeg,png}.
    """
    identities = sorted(
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    )
    for identity in identities:
        person_dir = os.path.join(data_root, identity)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(person_dir, ext)))
        for p in sorted(paths):
            yield identity, p


# ---------------------------------------------------------------------------
# Detection (InsightFace RetinaFace)
# ---------------------------------------------------------------------------

_detector = None  # lazy singleton, loaded once per process


def _get_detector():
    global _detector
    if _detector is None:
        from insightface.app import FaceAnalysis
        _detector = FaceAnalysis(name=config.INSIGHTFACE_DET_MODEL, allowed_modules=["detection"])
        _detector.prepare(ctx_id=config.DETECTOR_CTX_ID, det_size=(640, 640))
        logger.info("InsightFace detector (%s) loaded.", config.INSIGHTFACE_DET_MODEL)
    return _detector


def detect_faces(img_bgr: np.ndarray, image_path: str = "") -> List[FaceDetection]:
    """
    Returns ALL detected faces (an image could contain more than one),
    sorted by detection score, descending. Caller decides how to handle
    multi-face images.
    """
    app = _get_detector()
    faces = app.get(img_bgr)
    results = []
    for f in faces:
        results.append(
            FaceDetection(
                bbox=np.array(f.bbox, dtype=np.float32),
                landmarks=np.array(f.kps, dtype=np.float32),  # (5,2): InsightFace kps order
                det_score=float(f.det_score),
                image_path=image_path,
            )
        )
    results.sort(key=lambda fd: fd.det_score, reverse=True)
    return results


def largest_or_best_face(faces: List[FaceDetection]) -> Optional[FaceDetection]:
    """Convenience: pick the single best-scoring face, or None if no face found."""
    return faces[0] if faces else None


# ---------------------------------------------------------------------------
# Alignment — ArcFace only
# ---------------------------------------------------------------------------

# InsightFace's canonical 112x112 ArcFace template (standard 5-point target coords)
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(img_bgr: np.ndarray, face: FaceDetection, size: int = config.ARCFACE_INPUT_SIZE) -> np.ndarray:
    """
    Warps `img_bgr` so that the detected 5-point landmarks map onto ArcFace's
    canonical template, via an estimated similarity transform (rotation +
    uniform scale + translation, no shear). Returns a size x size BGR crop.
    """
    M, _ = cv2.estimateAffinePartial2D(face.landmarks, _ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Similarity transform estimation failed (degenerate landmarks).")
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)