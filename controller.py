# controller.py
"""
Stub implementation of the model controller.
Replace `answer_with_controller` with your real inference pipeline
(e.g., InstructBLIP + PPO gate + memory retrieval).
"""

from PIL import Image
from typing import Tuple

def answer_with_controller(
    image: Image.Image,
    question: str,
    source: str = "auto",
    distilled_model: str = "auto",
) -> Tuple[str, str, int]:
    """
    Returns:
        pred (str): predicted answer
        strategy_name (str): chosen strategy
        action_id (int): numeric ID of strategy
    """
    # --- Dummy logic ---
    # Always returns "Demo placeholder answer" with baseline strategy
    return "Demo placeholder answer", "baseline", 0
