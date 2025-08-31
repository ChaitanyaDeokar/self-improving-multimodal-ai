# controller.py — real VQA inference using BLIP (small, fast, no extra weights)
# Works on CPU Space. Uses HF Hub to download the model at first run.

import os
import torch
from PIL import Image
from typing import Tuple

from transformers import BlipForQuestionAnswering, BlipProcessor

# ---------------------------
# Load once at import time
# ---------------------------
HF_MODEL = os.getenv("HF_VQA_MODEL", "Salesforce/blip-vqa-base")  # small & good
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None

def _load():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(HF_MODEL)
        _model = BlipForQuestionAnswering.from_pretrained(HF_MODEL)
        _model.to(DEVICE)
        _model.eval()

def _answer_baseline(image: Image.Image, question: str) -> str:
    _load()
    inputs = _processor(images=image, text=question, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=10)
    ans = _processor.decode(out[0], skip_special_tokens=True)
    return ans.strip()

# --- optional future hooks (no-ops for now, keep API stable) ---
def _answer_with_memory(image: Image.Image, question: str) -> str:
    # Plug your FAISS/RAG here; fallback to baseline for now
    return _answer_baseline(image, question)

def _gate_auto(image: Image.Image, question: str) -> Tuple[int, str]:
    # When PPO or distilled are wired, pick actions here. For now: baseline (0).
    return 0, "baseline"

def _gate_distilled(image: Image.Image, question: str) -> Tuple[int, str]:
    # TODO: call your distilled classifier; fallback to baseline
    return 0, "baseline"

def _gate_ppo(image: Image.Image, question: str) -> Tuple[int, str]:
    # TODO: call your PPO policy; fallback to baseline
    return 0, "baseline"

# ---------------------------
# Public API for app.py
# ---------------------------
def answer_with_controller(
    image: Image.Image,
    question: str,
    source: str = "auto",
    distilled_model: str = "auto",
) -> Tuple[str, str, int]:
    """
    Returns:
        pred (str): predicted answer
        strategy_name (str): chosen strategy name
        action_id (int): numeric action (0=baseline, 1=memory in future, etc.)
    """
    source = (source or "auto").lower()

    if source == "baseline":
        ans = _answer_baseline(image, question)
        return ans, "baseline", 0
    elif source == "distilled":
        aid, label = _gate_distilled(image, question)
    elif source == "ppo":
        aid, label = _gate_ppo(image, question)
    else:  # auto
        aid, label = _gate_auto(image, question)

    # route by action id (for now all paths use baseline until you wire memory)
    if aid == 1:
        ans = _answer_with_memory(image, question)
    else:
        ans = _answer_baseline(image, question)

    return ans, label, aid
