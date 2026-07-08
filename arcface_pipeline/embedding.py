"""
embeddings.py
=============
ArcFace-only extractor. Takes an already-aligned 112x112 crop (from
common.align_face) and returns a raw embedding vector; L2 normalization is
applied centrally via get_normalized_embedding, the only path embeddings
should go through before being stored or compared.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

import config

logger = logging.getLogger(__name__)


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / max(norm, eps)


class ArcFaceExtractor:
    def __init__(self, model_path: str = config.ARCFACE_ONNX_PATH):
        import onnxruntime as ort
        # CoreMLExecutionProvider accelerates on macOS (Apple Neural Engine / GPU) and is
        # a mature, well-supported onnxruntime backend (unlike PyTorch's MPS, which has
        # known operator-coverage and precision gaps). CPU is always included as a
        # guaranteed fallback if CoreML can't run a particular op.
        available = ort.get_available_providers()
        providers = [p for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider"] if p in available]
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ArcFace ONNX model loaded from %s using providers=%s", model_path, self.session.get_providers())

    def extract(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        # ArcFace ONNX models standardly expect RGB, CHW, float32, normalized to [-1, 1]
        img = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))      # HWC -> CHW
        img = np.expand_dims(img, axis=0)        # add batch dim
        out = self.session.run(None, {self.input_name: img})[0]
        return out.flatten()


def get_normalized_embedding(extractor: ArcFaceExtractor, aligned_crop: np.ndarray) -> np.ndarray:
    """Extract + L2 normalize in one call — the only path embeddings should go through."""
    raw = extractor.extract(aligned_crop)
    return l2_normalize(raw)