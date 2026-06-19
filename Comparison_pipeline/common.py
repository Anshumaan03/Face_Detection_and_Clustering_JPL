"""
common.py
=========
The ONE shared module for: loading images, detecting faces + 5-point landmarks,
and aligning each detected face according to a target model's expected input
convention.

Design intent
--------------
Detection runs ONCE per image (InsightFace RetinaFace, via buffalo_l pack).
Every embedding model then gets its OWN alignment branch, because ArcFace,
dlib-resnet, FaceNet, and SigLIP2 were each trained on differently-prepared
crops. Forcing one alignment convention onto all 4 models is the #1 reason
a previously-working model (e.g. ArcFace) can suddenly score much worse —
it's silently being fed out-of-distribution input.

Typical usage
-------------
    from common import load_image, detect_faces, align_face

    img = load_image(path)
    faces = detect_faces(img)              # list[FaceDetection]
    for face in faces:
        arcface_crop = align_face(img, face, model="arcface")
        dlib_crop    = align_face(img, face, model="dlib_resnet")
        facenet_crop = align_face(img, face, model="facenet")
        siglip_crop  = align_face(img, face, model="siglip2")
"""

from __future__ import annotations

import os
import glob
import logging
from dataclasses import dataclass, field
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
    """
    Loads an image as BGR uint8 (OpenCV convention), since InsightFace and
    dlib both expect BGR or handle conversion internally. Raises a clear
    error instead of silently returning None, since a silent bad-load was
    likely contributing to noisy results before.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read image (corrupt or unsupported): {path}")
    return img


def iter_dataset(data_root: str = config.DATA_ROOT):
    """
    Yields (identity_label, image_path) for every image under
    data_root/<identity>/*.{jpg,jpeg,png}. Matches your folder structure:
    data/raw/person1/*, data/raw/person2/*, ...
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
# Detection (InsightFace RetinaFace, shared by all 4 models)
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
    Runs detection once. Returns ALL detected faces (a dataset image could
    contain more than one face) sorted by detection score, descending.
    Caller decides how to handle multi-face images (e.g. take highest-score
    face, or skip image, or keep all) — this function stays unopinionated.
    """
    app = _get_detector()
    faces = app.get(img_bgr)
    results = []
    for f in faces:
        results.append(
            FaceDetection(
                bbox=np.array(f.bbox, dtype=np.float32),
                landmarks=np.array(f.kps, dtype=np.float32),  # (5,2): InsightFace gives kps in this order
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
# Alignment — per-model branches
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


def _similarity_transform_align(img_bgr: np.ndarray, landmarks: np.ndarray, template: np.ndarray, size: int) -> np.ndarray:
    """
    Warps `img_bgr` so that `landmarks` map onto `template`, via an estimated
    similarity transform (rotation + uniform scale + translation — no shear,
    which is what face alignment papers use). Output is size x size.
    """
    M, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Similarity transform estimation failed (degenerate landmarks).")
    aligned = cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)
    return aligned


def _bbox_margin_crop(img_bgr: np.ndarray, bbox: np.ndarray, margin_frac: float, out_size: int) -> np.ndarray:
    """
    Crops a square region around bbox, expanded by margin_frac on each side,
    clipped to image bounds, then resizes to out_size x out_size.
    No landmark warping — used for models trained on natural/looser crops.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # expand to square using the larger side, then apply margin
    side = max(bw, bh) * (1 + 2 * margin_frac)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(w, cx + side / 2))
    ny2 = int(min(h, cy + side / 2))

    crop = img_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        raise RuntimeError("Empty crop — bbox out of bounds or degenerate.")
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


# dlib alignment needs its own shape predictor (NOT the same landmark format
# as InsightFace's kps) — works with either the 5-pt or 68-pt dlib predictor;
# loaded lazily, only if dlib_resnet is used.
_dlib_detector = None
_dlib_shape_predictor = None


def _get_dlib_predictor():
    global _dlib_detector, _dlib_shape_predictor
    if _dlib_shape_predictor is None:
        import dlib
        _dlib_detector = dlib.get_frontal_face_detector()
        _dlib_shape_predictor = dlib.shape_predictor(config.DLIB_SHAPE_PREDICTOR_PATH)
        logger.info("dlib shape predictor loaded from %s.", config.DLIB_SHAPE_PREDICTOR_PATH)
    return _dlib_detector, _dlib_shape_predictor


def _dlib_chip_align(img_bgr: np.ndarray, bbox: np.ndarray, out_size: int) -> np.ndarray:
    """
    dlib has its own landmark format + chip extraction (get_face_chip), which
    expects a dlib.rectangle and runs its own shape predictor internally —
    it does NOT accept InsightFace's kps directly. We re-detect the face
    region with dlib's own shape predictor inside the InsightFace bbox crop
    region to get dlib-native landmarks, then let dlib build the chip.

    This is the one branch where two detectors are involved by necessity:
    dlib's recognition model was trained against dlib's own alignment, and
    mixing in foreign landmarks here would reintroduce the exact problem
    we're trying to avoid.
    """
    import dlib

    detector, predictor = _get_dlib_predictor()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape = predictor(img_rgb, rect)
    chip = dlib.get_face_chip(img_rgb, shape, size=out_size)
    chip_bgr = cv2.cvtColor(np.array(chip), cv2.COLOR_RGB2BGR)
    return chip_bgr


def align_face(img_bgr: np.ndarray, face: FaceDetection, model: str) -> np.ndarray:
    """
    Single entry point: given the ORIGINAL image and a FaceDetection (from
    detect_faces), returns a crop aligned according to `model`'s convention
    as defined in config.MODEL_INPUT_SPECS. Returned image is BGR uint8,
    size x size, ready for that model's own preprocessing (normalization,
    channel order, etc. — handled in embeddings.py, NOT here).
    """
    spec = config.MODEL_INPUT_SPECS[model]
    size = spec["size"]
    alignment = spec["alignment"]

    if alignment == "insightface_norm_crop":
        return _similarity_transform_align(img_bgr, face.landmarks, _ARCFACE_TEMPLATE_112, size)

    elif alignment == "dlib_chip":
        return _dlib_chip_align(img_bgr, face.bbox, size)

    elif alignment == "bbox_margin_crop":
        margin = spec.get("margin", 0.3)
        return _bbox_margin_crop(img_bgr, face.bbox, margin, size)

    else:
        raise ValueError(f"Unknown alignment strategy '{alignment}' for model '{model}'")


def align_all_models(img_bgr: np.ndarray, face: FaceDetection) -> dict:
    """Convenience: returns {model_name: aligned_crop} for every configured model."""
    return {m: align_face(img_bgr, face, m) for m in config.MODEL_INPUT_SPECS}