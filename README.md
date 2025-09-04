# 🧠 Self-Improving Multimodal AI
**Reinforcement Learning–powered Vision–Language Agent with Memory-Augmented Retrieval**

This project implements a **self-improving multimodal AI system** that answers visual questions about images with **high accuracy, low latency, and improved grounding**.  
It combines three core components:  

- 🖼️ **Vision–Language Model (InstructBLIP Flan-T5-XL)** – image–text grounding.  
- 📚 **Memory-Augmented Retrieval (MAR)** – recalls similar solved cases to improve consistency.  
- 🎯 **Reinforcement Learning (PPO Controller)** – learns when to use memory vs. answer directly, optimising the trade-off between **accuracy, faithfulness, and latency**.  

---

## 🚀 Key Features
- **Adaptive Retrieval** – dynamically decides per question whether to consult memory.  
- **Accuracy & Speed** – achieves **81.4% Strict accuracy, 75.6% VQA2 accuracy** within ~392 ms latency.  
- **Faithfulness** – selectively reuses memory while avoiding distractors, improving grounding of answers.  
- **Deployment Ready** – distilled lightweight gate for real-time applications.  
- **Interactive Demo** – try it live on [Hugging Face Spaces](https://huggingface.co/spaces/cd2412/VQA_Memory__RL_Controller).  

---

## 📊 Results (Validation on 1,000 items)
| Method        | Strict Accuracy | VQA2 Accuracy | Mean Latency | p95 Latency |
|---------------|----------------|---------------|--------------|-------------|
| Direct-only   | 81.10%         | 75.37%        | ~250 ms      | ~400 ms     |
| MAR-always    | 79.40%         | 74.13%        | ~650 ms      | ~1400 ms    |
| **PPO (final)** | **81.40%**     | **75.60%**    | **392 ms**   | **1080 ms** |

---

## ⚙️ Tech Stack
- **Python** (3.10+)  
- **PyTorch** – deep learning framework  
- **Hugging Face Transformers** – vision–language backbone  
- **FAISS** – memory-augmented retrieval  
- **Stable-Baselines3 (PPO)** – reinforcement learning controller  
- **Gradio** – interactive web demo  
- **NumPy / Pandas / Matplotlib** – data handling & analysis  

---

## 📂 Repository Structure
self-improving-multimodal-ai/
│── notebooks/ # Training & evaluation notebooks
│── models/ # Saved PPO checkpoints & distilled models
│── data/ # Splits & configs (no raw datasets)
│── src/ # Core pipeline (encoder, retriever, gate, generator)
│── demo/ # Gradio app for interactive demo
│── README.md # Project documentation


---

## ▶️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/self-improving-multimodal-ai.git
   cd self-improving-multimodal-ai
   
2. Install dependencies:
   pip install -r requirements.txt
   
4. Launch demo locally:
   python app.py

5.Or try it directly on Hugging Face Spaces

## 📌 Use Cases
Product Support AI – answering customer queries about products from images.

Education & Tutoring – assisting with visual reasoning tasks.

Healthcare Support – retrieval-augmented reasoning over medical imagery (future work).

Interactive Assistants – grounding responses with both images and context.

## ✨ Contributions & Future Work

Add spatial priors for “where” questions.

Improve faithfulness logging with large-scale attribution checks.

Extend to multi-turn dialogues and domain-specific datasets.

Explore cost-aware reward shaping for production.

## 👨‍💻 Author

Chaitanya Rajesh Deokar
MSc in Computing (Data Science) – Technological University Dublin
