# AI Sales Tools — Hackathon Project

Two AI-powered tools built to enhance sales and customer interactions.

---

## Challenge 1: The Empathy Engine 🎙

A Text-to-Speech service that detects the emotion in any text and speaks it back with a matching voice — faster and higher for joy, slower and quieter for sadness, and so on.

**How to run:**
```bash
cd empathy-engine
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

---

## Challenge 2: The Pitch Visualizer 🖼

A storyboard generator that takes a sales narrative, breaks it into scenes, and generates an AI image for each panel using Hugging Face.

**Setup:**
1. Create a `.env` file inside the `pitch-visualizer` folder:
   ```
   HF_API_KEY=hf_your_token_here
   ```
2. Make sure your Hugging Face token has **"Make calls to Inference Providers"** enabled.

**How to run:**
```bash
cd pitch-visualizer
pip install -r requirements.txt
python app.py
```
Open http://localhost:5001

---

## Tech Stack

- Python, Flask
- Hugging Face (image generation)
- Transformers / VADER (emotion detection)
- pyttsx3 / gTTS (text-to-speech)
- NLTK (text segmentation)
