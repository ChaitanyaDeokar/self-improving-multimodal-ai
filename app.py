# app.py
"""
VQA — Memory + RL Controller (Gradio app)
- Drag-and-drop an image, ask a question, and see the model's answer + chosen strategy.
- Tries to import `answer_with_controller` from controller.py. Falls back to a stub if missing.
- Works on Hugging Face Spaces, Render, Docker, or local run.
"""

import os
import sys
import time
import traceback
import subprocess
from typing import Tuple, Optional

# Ensure gradio is available when running locally; Spaces installs from requirements.txt
try:
    import gradio as gr
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

from PIL import Image

# -----------------------------
# Attempt to import real handler
# -----------------------------
def _make_fallback():
    def _fallback_answer_with_controller(
        image: Image.Image,
        question: str,
        source: str = "auto",
        distilled_model: str = "auto",
    ) -> Tuple[str, str, int]:
        # Replace with real inference to remove this placeholder.
        return "Placeholder answer (wire your models in controller.py).", "baseline", 0
    return _fallback_answer_with_controller

try:
    # Expect controller.py to define: answer_with_controller(image, question, source, distilled_model)
    from controller import answer_with_controller  # type: ignore
except Exception as e:
    print(f"[WARN] Using fallback controller because import failed: {e}", flush=True)
    answer_with_controller = _make_fallback()

# -----------------------------
# UI Constants
# -----------------------------
TITLE = "VQA — Memory + RL Controller"
DESCRIPTION = (
    "Upload an image, enter a question, and the controller will choose the best decoding strategy."
)

CONTROLLER_SOURCES = ["auto", "distilled", "ppo", "baseline"]
DISTILLED_CHOICES = ["auto", "logreg", "mlp32"]

# -----------------------------
# Inference wrapper with guards
# -----------------------------
def vqa_demo_fn(
    image: Optional[Image.Image],
    question: str,
    source: str,
    distilled_model: str,
) -> Tuple[str, str, float]:
    """Safely run inference and return (answer, strategy_label, latency_ms)."""
    # Input validation
    if image is None:
        return "Please upload an image.", "", 0.0
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", 0.0

    # Convert & measure latency
    t0 = time.perf_counter()
    try:
        # Convert to RGB to avoid issues with PNG/L mode
        image_rgb = image.convert("RGB")

        pred, strategy_name, action_id = answer_with_controller(
            image_rgb,
            question,
            source=source,
            distilled_model=distilled_model,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Friendly formatting
        strategy_out = f"{action_id} → {strategy_name}"
        return str(pred), strategy_out, round(latency_ms, 1)

    except Exception as err:
        # Never crash the app — show a concise error to the user and log details to server
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print("[ERROR] Inference failed:\n" + "".join(traceback.format_exc()), flush=True)
        return f"Error: {err}", "error", round(latency_ms, 1)

# -----------------------------
# Build Gradio Interface
# -----------------------------
with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
    gr.Markdown(f"### {TITLE}\n{DESCRIPTION}")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(
                type="pil",
                label="Image",
                height=320,
                sources=["upload", "drag-and-drop", "clipboard", "webcam"],
                image_mode="RGB",
            )
            q_in = gr.Textbox(
                label="Question",
                placeholder="e.g., What colour is the bus?",
                lines=2,
                max_lines=4,
            )
            source_in = gr.Radio(
                CONTROLLER_SOURCES,
                value="auto",
                label="Controller Source",
            )
            dist_in = gr.Radio(
                DISTILLED_CHOICES,
                value="auto",
                label="Distilled Gate (if used)",
            )
            run_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            ans_out = gr.Textbox(label="Answer", interactive=False, lines=3, max_lines=6)
            strat_out = gr.Textbox(label="Chosen Strategy", interactive=False)
            lat_out = gr.Number(label="Latency (ms)", precision=1, interactive=False)

    run_btn.click(
        vqa_demo_fn,
        inputs=[img_in, q_in, source_in, dist_in],
        outputs=[ans_out, strat_out, lat_out],
        api_name="predict",
    )

# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    # Respect $PORT for Spaces/Render/Docker; default to 7860 locally
    port = int(os.getenv("PORT", "7860"))
    # Queue improves robustness under load
    demo.queue(concurrency_count=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,          # set True only for local quick sharing
        show_error=True,
    )
