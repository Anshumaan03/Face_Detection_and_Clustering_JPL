"""
embeddings.py
=============
One extractor class per model, each taking an ALREADY-ALIGNED crop (from
common.align_face) and returning a raw embedding vector. Model-specific
preprocessing (channel order, normalization, mean/std) lives here, since
it's specific to each network's training recipe — alignment (common.py)
and the embedding extraction (this file) are deliberately kept separate
concerns.

All extractors expose the same interface:

    extractor = ArcFaceExtractor()
    vec = extractor.extract(aligned_bgr_crop)   # -> np.ndarray, raw (unnormalized)

L2 normalization is applied centrally in `get_normalized_embedding`, AFTER
extraction, so every model's output is comparable in the same way before
HDBSCAN. Do this consistently — mixing normalized and raw embeddings across
models is another classic silent-degradation source.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared normalization — applied AFTER extraction, identically for all models
# ---------------------------------------------------------------------------

def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / max(norm, eps)


# ---------------------------------------------------------------------------
# ArcFace (.onnx, via onnxruntime)
# ---------------------------------------------------------------------------

class ArcFaceExtractor:
    name = "arcface"

    def __init__(self, model_path: str = config.ARCFACE_ONNX_PATH):
        import onnxruntime as ort
        # CoreMLExecutionProvider accelerates on macOS (Apple Neural Engine / GPU) and is
        # a mature, well-supported onnxruntime backend (unlike PyTorch's MPS, which has
        # known operator-coverage and precision gaps) — safe to use here. CPU is always
        # included as a guaranteed fallback if CoreML can't run a particular op.
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


# ---------------------------------------------------------------------------
# dlib ResNet (dlib_face_recognition_resnet_model_v1.dat)
# ---------------------------------------------------------------------------

class DlibResnetExtractor:
    name = "dlib_resnet"

    def __init__(self, model_path: str = config.DLIB_RESNET_PATH):
        import dlib
        self.model = dlib.face_recognition_model_v1(model_path)
        logger.info("dlib ResNet recognition model loaded from %s", model_path)

    def extract(self, aligned_bgr_150: np.ndarray) -> np.ndarray:
        # dlib's compute_face_descriptor expects RGB
        img_rgb = cv2.cvtColor(aligned_bgr_150, cv2.COLOR_BGR2RGB)
        descriptor = self.model.compute_face_descriptor(img_rgb)
        return np.array(descriptor, dtype=np.float32)


# ---------------------------------------------------------------------------
# FaceNet (facenet_vggface2.pt, via facenet-pytorch's InceptionResnetV1 arch)
# ---------------------------------------------------------------------------

class FaceNetExtractor:
    name = "facenet"

    def __init__(self, model_path: str = config.FACENET_VGGFACE2_PATH, device: str = None):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained=None, classify=False)
        state_dict = torch.load(model_path, map_location=self.device)
        # Checkpoint includes a final classification head (logits.weight/bias) from
        # whatever identity-classification task it was trained/fine-tuned on. We only
        # want the embedding trunk, so drop those keys before loading — classify=False
        # above means our model has no `logits` layer to receive them anyway.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logits.")}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            logger.warning("FaceNet checkpoint had unexpected keys not loaded: %s", unexpected)
        if missing:
            logger.warning("FaceNet model has uninitialized keys not found in checkpoint: %s", missing)
        self.model.to(self.device).eval()
        self.torch = torch
        logger.info("FaceNet (vggface2) loaded from %s on %s", model_path, self.device)

    def extract(self, aligned_bgr_160: np.ndarray) -> np.ndarray:
        # facenet-pytorch expects RGB, CHW, float32 scaled to roughly [-1, 1]
        img = cv2.cvtColor(aligned_bgr_160, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        tensor = self.torch.tensor(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.model(tensor)
        return out.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# SigLIP2 (google/siglip2-base-patch16-224, via transformers)
# ---------------------------------------------------------------------------

class SigLIP2Extractor:
    name = "siglip2"

    def __init__(self, model_name: str = config.SIGLIP2_MODEL_NAME, device: str = None):
        import torch
        from transformers import AutoModel, AutoImageProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.torch = torch
        logger.info("SigLIP2 (%s) loaded on %s", model_name, self.device)

    def extract(self, aligned_bgr_224: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(aligned_bgr_224, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            # use the vision tower's pooled output as the embedding
            out = self.model.get_image_features(**inputs)
        # Depending on the installed transformers version, get_image_features() may
        # return either a raw tensor or a BaseModelOutputWithPooling wrapper object
        # (newer versions). Handle both so this doesn't break on a version bump.
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        return out.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXTRACTOR_CLASSES = {
    "arcface": ArcFaceExtractor,
    "dlib_resnet": DlibResnetExtractor,
    "facenet": FaceNetExtractor,
    "siglip2": SigLIP2Extractor,
}


def load_all_extractors() -> dict:
    """Instantiates all 4 extractors once. Call this once per process, reuse for every image."""
    return {name: cls() for name, cls in EXTRACTOR_CLASSES.items()}


def get_normalized_embedding(extractor, aligned_crop: np.ndarray) -> np.ndarray:
    """Extract + L2 normalize in one call — the only path embeddings should go through."""
    raw = extractor.extract(aligned_crop)
    return l2_normalize(raw)