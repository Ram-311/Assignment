"""
Empathy Engine - AI-powered emotionally expressive Text-to-Speech
"""

import os
import io
import re
import json
import uuid
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from flask import Flask, request, jsonify, send_file, render_template_string
from transformers import pipeline

# ── Emotion Detection ─────────────────────────────────────────────────────────

# We use a fine-tuned BERT-based model for multi-class emotion classification.
# Falls back to VADER-based heuristics if the model can't be loaded.
try:
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )
    USE_TRANSFORMER = True
    print("[✓] Transformer emotion model loaded.")
except Exception as e:
    print(f"[!] Transformer model unavailable ({e}). Using VADER fallback.")
    USE_TRANSFORMER = False
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()


# ── Emotion → Voice Parameter Mapping ────────────────────────────────────────

@dataclass
class VoiceProfile:
    rate: int          # words per minute  (pyttsx3 range ~80-300)
    pitch_factor: float  # relative: 1.0 = neutral, >1 higher, <1 lower
    volume: float      # 0.0 – 1.0
    label: str
    description: str


EMOTION_PROFILES: dict[str, VoiceProfile] = {
    "joy": VoiceProfile(
        rate=185, pitch_factor=1.25, volume=0.95,
        label="Joy 😊",
        description="Faster, higher pitch, full volume – bright and energetic"
    ),
    "surprise": VoiceProfile(
        rate=200, pitch_factor=1.35, volume=1.0,
        label="Surprise 😮",
        description="Fast, noticeably high pitch – sudden exclamation"
    ),
    "anger": VoiceProfile(
        rate=160, pitch_factor=0.88, volume=1.0,
        label="Anger 😠",
        description="Moderate pace, lower pitch, maximum volume – forceful"
    ),
    "fear": VoiceProfile(
        rate=210, pitch_factor=1.15, volume=0.70,
        label="Fear 😨",
        description="Rapid, slightly high pitch, quiet – tense & hushed"
    ),
    "sadness": VoiceProfile(
        rate=115, pitch_factor=0.80, volume=0.65,
        label="Sadness 😢",
        description="Slow, low pitch, subdued volume – heavy & sombre"
    ),
    "disgust": VoiceProfile(
        rate=130, pitch_factor=0.85, volume=0.85,
        label="Disgust 🤢",
        description="Slow, low pitch – flat and dismissive"
    ),
    "neutral": VoiceProfile(
        rate=150, pitch_factor=1.0, volume=0.80,
        label="Neutral 😐",
        description="Balanced, conversational delivery"
    ),
    # Bonus: inquisitive (mapped from a weak-confidence blend)
    "inquisitive": VoiceProfile(
        rate=160, pitch_factor=1.18, volume=0.82,
        label="Inquisitive 🤔",
        description="Medium pace, rising pitch – curious and thoughtful"
    ),
}

# Intensity scaling: score 0-1 → scale factor applied on top of base profile
def _intensity_scale(base_value: float, neutral: float, score: float, scale_range: float = 0.20) -> float:
    """Linearly scale the deviation from neutral by the emotion score."""
    deviation = base_value - neutral
    return neutral + deviation * (0.5 + score * 0.5 + (score - 0.5) * scale_range)


# -------------> Emotion Detection Logic <--------------------------------------------

def detect_emotion(text: str) -> tuple[str, float, list[dict]]:
    """
    Returns (emotion_label, confidence_score, all_scores).
    Includes a heuristic override for questions → 'inquisitive'.
    """
    # ---------------> Question heuristic (bonus: inquisitive state) <---------------------
    stripped = text.strip()
    if stripped.endswith("?") or re.search(r'\b(why|what|how|when|where|who|which|whose|whom)\b', stripped, re.I):
        if USE_TRANSFORMER:
            raw = emotion_classifier(text)[0]
            all_scores = [{"label": r["label"], "score": round(r["score"], 4)} for r in raw]
            top = max(raw, key=lambda x: x["score"])
            # -----------------> Only override if model isn't very confident about a specific emotion <--------------------
            if top["score"] < 0.55:
                return "inquisitive", 0.70, all_scores
        else:
            return "inquisitive", 0.70, []

    if USE_TRANSFORMER:
        raw = emotion_classifier(text)[0]
        all_scores = [{"label": r["label"], "score": round(r["score"], 4)} for r in raw]
        top = max(raw, key=lambda x: x["score"])
        label = top["label"].lower()
        if label not in EMOTION_PROFILES:
            label = "neutral"
        return label, round(top["score"], 4), all_scores
    else:
        # -----------> VADER fallback → maps compound to joy/sadness/neutral <------------------
        scores = vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.5:
            return "joy", min(compound, 1.0), [scores]
        elif compound <= -0.5:
            return "sadness", min(abs(compound), 1.0), [scores]
        else:
            return "neutral", 1 - abs(compound), [scores]


# --------------> TTS with pyttsx3 <-----------------------------

def synthesize(text: str, profile: VoiceProfile, intensity: float, output_path: str) -> str:
    """Generate a WAV file using pyttsx3 with modulated voice parameters."""
    import pyttsx3

    engine = pyttsx3.init()

    # -------------------> Apply intensity scaling on rate and pitch-factor <------------------------
    scaled_rate = int(_intensity_scale(profile.rate, 150, intensity))
    scaled_volume = float(_intensity_scale(profile.volume, 0.80, intensity))
    scaled_volume = max(0.0, min(1.0, scaled_volume))

    engine.setProperty("rate", scaled_rate)
    engine.setProperty("volume", scaled_volume)

    try:
        voices = engine.getProperty("voices")
        
        if profile.pitch_factor > 1.0 and len(voices) > 1:
            engine.setProperty("voice", voices[1].id)
        else:
            engine.setProperty("voice", voices[0].id)
    except Exception:
        pass

    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    return output_path


# ----------------------> gTTS fallback (online) <------------------------

def synthesize_gtts(text: str, profile: VoiceProfile, output_path: str) -> str:
    """Fallback to gTTS when pyttsx3 is unavailable."""
    from gtts import gTTS
    slow = profile.rate < 130
    tts = gTTS(text=text, lang="en", slow=slow)
    tts.save(output_path)
    return output_path


# ----------------------->  Flask App <----------------------------------------

app = Flask(__name__)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Empathy Engine</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0d0f14;
      --surface: #161920;
      --surface2: #1e2230;
      --border: rgba(255,255,255,0.07);
      --accent: #c8f060;
      --accent2: #60d4f0;
      --text: #e8eaf0;
      --muted: #7a7f95;
      --radius: 16px;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      background-image:
        radial-gradient(ellipse 60% 40% at 70% 20%, rgba(200,240,96,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 20% 80%, rgba(96,212,240,0.06) 0%, transparent 60%);
    }

    .container {
      width: 100%;
      max-width: 760px;
    }

    header {
      margin-bottom: 3rem;
    }

    .eyebrow {
      font-size: 0.72rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 500;
      margin-bottom: 0.6rem;
    }

    h1 {
      font-family: 'DM Serif Display', serif;
      font-size: clamp(2.2rem, 5vw, 3.4rem);
      line-height: 1.1;
      font-style: italic;
      background: linear-gradient(135deg, var(--text) 40%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .subtitle {
      margin-top: 0.75rem;
      color: var(--muted);
      font-size: 0.95rem;
      font-weight: 300;
      line-height: 1.6;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      margin-bottom: 1.25rem;
    }

    label {
      display: block;
      font-size: 0.8rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 500;
      margin-bottom: 0.6rem;
    }

    textarea {
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
      font-size: 1rem;
      line-height: 1.6;
      padding: 1rem;
      resize: vertical;
      min-height: 130px;
      transition: border-color 0.2s;
      outline: none;
    }

    textarea:focus {
      border-color: rgba(200,240,96,0.4);
    }

    textarea::placeholder { color: var(--muted); }

    button {
      width: 100%;
      margin-top: 1.25rem;
      padding: 1rem 2rem;
      background: var(--accent);
      color: #0d0f14;
      border: none;
      border-radius: 10px;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.15s;
    }

    button:hover { opacity: 0.88; transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    #result { display: none; }

    .emotion-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.35rem 0.9rem;
      font-size: 0.88rem;
      font-weight: 500;
      margin-bottom: 1rem;
    }

    .params-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }

    .param {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.9rem;
      text-align: center;
    }

    .param-val {
      font-family: 'DM Serif Display', serif;
      font-size: 1.6rem;
      color: var(--accent);
    }

    .param-name {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      margin-top: 0.2rem;
    }

    audio {
      width: 100%;
      margin-top: 0.5rem;
      border-radius: 8px;
      outline: none;
    }

    .scores {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
    }

    .score-pill {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.25rem 0.6rem;
      font-size: 0.78rem;
      color: var(--muted);
    }

    .score-pill span { color: var(--text); font-weight: 500; }

    .spinner {
      display: none;
      width: 18px; height: 18px;
      border: 2px solid rgba(13,15,20,0.3);
      border-top-color: #0d0f14;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin: 0 auto;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .error-msg {
      color: #ff7070;
      font-size: 0.88rem;
      margin-top: 0.75rem;
      display: none;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="eyebrow">AI Voice Lab</div>
    <h1>The Empathy Engine</h1>
    <p class="subtitle">Type anything. The engine detects your emotion and speaks it back with a matching voice.</p>
  </header>

  <div class="card">
    <label for="inputText">Your text</label>
    <textarea id="inputText" placeholder="e.g. I just got the promotion I've been working towards for two years!"></textarea>
    <button id="synthBtn" onclick="synthesize()">
      <span id="btnLabel">Synthesize Voice</span>
      <div class="spinner" id="spinner"></div>
    </button>
    <div class="error-msg" id="errMsg"></div>
  </div>

  <div class="card" id="result">
    <label>Detected Emotion</label>
    <div class="emotion-badge" id="emotionBadge"></div>
    <p id="emotionDesc" style="color:var(--muted);font-size:0.88rem;margin-bottom:1.2rem;"></p>

    <label>Voice Parameters</label>
    <div class="params-grid">
      <div class="param"><div class="param-val" id="pRate">—</div><div class="param-name">Rate (wpm)</div></div>
      <div class="param"><div class="param-val" id="pPitch">—</div><div class="param-name">Pitch Factor</div></div>
      <div class="param"><div class="param-val" id="pVol">—</div><div class="param-name">Volume</div></div>
    </div>

    <label>Audio Output</label>
    <audio id="audioPlayer" controls></audio>

    <div class="scores" id="scoresContainer"></div>
  </div>
</div>

<script>
async function synthesize() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) return;

  const btn = document.getElementById('synthBtn');
  const label = document.getElementById('btnLabel');
  const spinner = document.getElementById('spinner');
  const err = document.getElementById('errMsg');

  btn.disabled = true;
  label.style.display = 'none';
  spinner.style.display = 'block';
  err.style.display = 'none';
  document.getElementById('result').style.display = 'none';

  try {
    const res = await fetch('/synthesize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Unknown error');

    document.getElementById('emotionBadge').textContent = data.emotion_label;
    document.getElementById('emotionDesc').textContent = data.description;
    document.getElementById('pRate').textContent = data.rate;
    document.getElementById('pPitch').textContent = data.pitch_factor.toFixed(2) + '×';
    document.getElementById('pVol').textContent = Math.round(data.volume * 100) + '%';

    const audio = document.getElementById('audioPlayer');
    audio.src = '/audio/' + data.audio_file + '?t=' + Date.now();
    audio.load();

    // Render emotion scores
    const sc = document.getElementById('scoresContainer');
    sc.innerHTML = '';
    if (data.all_scores && data.all_scores.length) {
      data.all_scores.sort((a,b) => b.score - a.score).forEach(s => {
        sc.innerHTML += `<div class="score-pill">${s.label}: <span>${(s.score*100).toFixed(1)}%</span></div>`;
      });
    }

    document.getElementById('result').style.display = 'block';
  } catch(e) {
    err.textContent = '⚠ ' + e.message;
    err.style.display = 'block';
  } finally {
    btn.disabled = false;
    label.style.display = 'block';
    spinner.style.display = 'none';
  }
}

document.getElementById('inputText').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) synthesize();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/synthesize", methods=["POST"])
def synthesize_route():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    emotion, confidence, all_scores = detect_emotion(text)
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = str(OUTPUT_DIR / filename)

    # Try pyttsx3 first, fall back to gTTS
    try:
        synthesize(text, profile, confidence, output_path.replace(".mp3", ".wav"))
        output_path = output_path.replace(".mp3", ".wav")
        filename = filename.replace(".mp3", ".wav")
    except Exception as e1:
        try:
            synthesize_gtts(text, profile, output_path)
        except Exception as e2:
            return jsonify({"error": f"TTS failed: {e1} / {e2}"}), 500

    return jsonify({
        "emotion": emotion,
        "emotion_label": profile.label,
        "description": profile.description,
        "confidence": confidence,
        "rate": profile.rate,
        "pitch_factor": profile.pitch_factor,
        "volume": profile.volume,
        "audio_file": filename,
        "all_scores": all_scores,
    })


@app.route("/audio/<filename>")
def audio_file(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    mime = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return send_file(str(path), mimetype=mime)


# ----------------------------> CLI Mode <-----------------------------------------------

def cli_mode():
    import sys
    print("\n=== Empathy Engine (CLI) ===")
    text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Enter text: ").strip()
    if not text:
        print("No text provided."); return

    emotion, confidence, all_scores = detect_emotion(text)
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    print(f"\n  Emotion   : {profile.label} (confidence: {confidence:.2%})")
    print(f"  Rate      : {profile.rate} wpm")
    print(f"  Pitch     : {profile.pitch_factor:.2f}×")
    print(f"  Volume    : {profile.volume:.0%}")
    print(f"  Desc      : {profile.description}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = str(OUTPUT_DIR / f"output_{emotion}.wav")
    try:
        synthesize(text, profile, confidence, output_path)
        print(f"\n  ✓ Audio saved → {output_path}")
    except Exception as e:
        mp3_path = output_path.replace(".wav", ".mp3")
        try:
            synthesize_gtts(text, profile, mp3_path)
            print(f"\n  ✓ Audio saved → {mp3_path}")
        except Exception as e2:
            print(f"\n  ✗ TTS error: {e} / {e2}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_mode()
    else:
        app.run(debug=True, port=5000)