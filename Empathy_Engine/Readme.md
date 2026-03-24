# 🎙 The Empathy Engine

> AI-powered Text-to-Speech that **dynamically modulates vocal characteristics** based on the detected emotion of the input text.

---

## What It Does

Standard TTS sounds robotic because it ignores emotional context. The Empathy Engine fixes this by:

1. **Detecting emotion** in your text using a fine-tuned DistilRoBERTa transformer model (7 emotions + "inquisitive" heuristic)
2. **Mapping that emotion** to a voice profile with specific `rate`, `pitch`, and `volume` parameters
3. **Scaling the parameters** by the emotion's confidence score (intensity scaling — stronger emotion = more extreme modulation)
4. **Synthesizing audio** via `pyttsx3` (offline) with `gTTS` as an online fallback
5. **Serving it** via a polished web UI or CLI

---

## Emotion → Voice Mapping

| Emotion | Rate (wpm) | Pitch Factor | Volume | Logic |
|---|---|---|---|---|
| Joy 😊 | 185 | 1.25× | 95% | Faster, higher, bright |
| Surprise 😮 | 200 | 1.35× | 100% | Very fast, high — exclamatory |
| Anger 😠 | 160 | 0.88× | 100% | Forceful, low, loud |
| Fear 😨 | 210 | 1.15× | 70% | Rapid, tense, hushed |
| Sadness 😢 | 115 | 0.80× | 65% | Slow, low, subdued |
| Disgust 🤢 | 130 | 0.85× | 85% | Flat, dismissive |
| Neutral 😐 | 150 | 1.00× | 80% | Balanced, conversational |
| Inquisitive 🤔 | 160 | 1.18× | 82% | Rising pitch, thoughtful |

**Intensity Scaling**: A high-confidence emotion amplifies the deviation from neutral. `"This is good."` → slight pitch up. `"This is the BEST NEWS EVER!!"` → much stronger modulation.

**Inquisitive Heuristic (Bonus)**: Sentences ending in `?` or starting with question words (`why`, `what`, `how`, etc.) are routed to the `inquisitive` profile if the model confidence is below 55%.

---

## Setup

### Prerequisites
- Python 3.10+
- `espeak` or `espeak-ng` installed (required by pyttsx3 on Linux):
  ```bash
  sudo apt-get install espeak-ng   # Ubuntu/Debian
  brew install espeak               # macOS
  ```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/empathy-engine.git
cd empathy-engine

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note**: The first run will download the `j-hartmann/emotion-english-distilroberta-base` model (~330 MB) from Hugging Face. This is a one-time download.

---

## Usage

### Web Interface (recommended)

```bash
python app.py
```

Open **http://localhost:5000** in your browser. Type any text and click "Synthesize Voice" (or press `Ctrl+Enter`). The page displays the detected emotion, all confidence scores, the exact parameter values used, and an embedded audio player.

### CLI Mode

```bash
# Interactive prompt
python app.py cli

# Pass text directly
python app.py cli "I just found out I got the job! I can't believe it!"
```

Generated audio is saved to the `outputs/` directory.

---

## Design Choices

### Emotion Model
We chose `j-hartmann/emotion-english-distilroberta-base` over simpler VADER/TextBlob because:
- It classifies 7 distinct emotions (joy, surprise, anger, fear, sadness, disgust, neutral) vs. VADER's positive/negative/neutral
- It returns confidence scores for all classes, enabling intensity scaling
- VADER is still included as a fallback for environments where `transformers`/`torch` can't be installed

### TTS Engine
`pyttsx3` is used as the primary engine because:
- Fully offline — no API key, no latency, no cost
- Exposes `rate` and `volume` programmatically
- Pitch is controlled indirectly via voice gender selection (female = higher perceived pitch)

`gTTS` is used as a fallback (requires internet) when pyttsx3 fails (e.g., no `espeak` installed).

### Intensity Scaling
The deviation of each parameter from its neutral value is multiplied by `0.5 + score * 0.5`, where `score` is the top emotion confidence (0–1). This means:
- Score = 0.5 → 75% of the full modulation applied
- Score = 1.0 → 100% of the full modulation applied
- Score = 0.0 → only 50% of the modulation (never fully collapses to neutral)

---

## Project Structure

```
empathy-engine/
├── app.py              # Main application (Flask + CLI)
├── requirements.txt
├── outputs/            # Generated audio files (created at runtime)
└── README.md
```

---

## Bonus Features Implemented

- ✅ **Granular Emotions**: 7 transformer-classified emotions + inquisitive heuristic
- ✅ **Intensity Scaling**: confidence score linearly scales parameter deviation from neutral
- ✅ **Web Interface**: Flask app with real-time emotion breakdown and audio player
- ✅ **Fallback chain**: pyttsx3 (offline) → gTTS (online) → error with clear message