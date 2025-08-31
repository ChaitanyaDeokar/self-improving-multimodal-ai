# app.py (fixed)
import os, sys, time, traceback, subprocess
from typing import Tuple, Optional
from PIL import Image

try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

def _make_fallback():
    def _fallback_answer_with_controller(image, question, source="auto", distilled_model="auto"):
        return "Placeholder answer (wire your models in controller.py).", "baseline", 0
    return _fallback_answer_with_controller

try:
    from controller import answer_with_controller
except Exception as e:
    print(f"[WARN] Using fallback controller because import failed: {e}", flush=True)
    answer_with_controller = _make_fallback()

TITLE = "VQA — Memory + RL Controller"
DESCRIPTION = "Upload an image, enter a question, and the controller will choose the best decoding strategy."

CONTROLLER_SOURCES = ["auto", "distilled", "ppo", "baseline"]
DISTILLED_CHOICES = ["auto", "logreg", "mlp32"]

def vqa_demo_fn(image: Optional[Image.Image], question: str, source: str, distilled_model: str) -> Tuple[str, str, float]:
    if image is None:
        return "Please upload an image.", "", 0.0
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", 0.0
    t0 = time.perf_counter()
    try:
        image_rgb = image.convert("RGB")
        pred, strategy_name, action_id = answer_with_controller(
            image_rgb, question, source=source, distilled_model=distilled_model
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return str(pred), f"{action_id} → {strategy_name}", round(latency_ms, 1)
    except Exception as err:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print("[ERROR] Inference failed:\n" + "".join(traceback.format_exc()), flush=True)
        return f"Error: {err}", "error", round(latency_ms, 1)

with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
    gr.Markdown(f"### {TITLE}\n{DESCRIPTION}")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(
                type="pil",
                label="Image",
                height=320,
                sources=["upload", "webcam", "clipboard"],  # ✅ fixed
            )
            q_in = gr.Textbox(label="Question", placeholder="e.g., What colour is the bus?", lines=2, max_lines=4)
            source_in = gr.Radio(CONTROLLER_SOURCES, value="auto", label="Controller Source")
            dist_in = gr.Radio(DISTILLED_CHOICES, value="auto", label="Distilled Gate (if used)")
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(concurrency_count=2)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True)
