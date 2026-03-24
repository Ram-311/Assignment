import os
import re
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found. Please set it in the .env file.")

print("HF API Key loaded successfully!")

OUTPUT_DIR = Path("static/storyboards")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_STYLES = {
    "cinematic":   "cinematic film still, dramatic lighting, shallow depth of field, ultra realistic, 8k",
    "digital_art": "digital concept art, vibrant colors, detailed illustration, trending on ArtStation",
    "watercolor":  "delicate watercolor illustration, soft washes, organic textures, painterly",
    "corporate":   "clean professional photography, bright office environment, business lifestyle stock photo",
    "comic":       "graphic novel panel, bold ink outlines, flat cel-shaded colors, dynamic composition",
    "oil_painting":"classical oil painting style, rich impasto texture, dramatic chiaroscuro, Old Masters",
}

# ── Narrative Segmentation ────────────────────────────────────────────────────

def segment_text(text: str) -> list[str]:
    """Split text into 3-6 logical scene panels."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text.strip())
    except ImportError:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]

    # Merge very short sentences with the next one
    merged: list[str] = []
    buffer = ""
    for s in sentences:
        if buffer:
            combined = buffer + " " + s
            if len(combined.split()) < 12:
                buffer = combined
            else:
                merged.append(buffer)
                buffer = s
        else:
            if len(s.split()) < 6 and sentences.index(s) < len(sentences) - 1:
                buffer = s
            else:
                merged.append(s)
    if buffer:
        merged.append(buffer)

    if len(merged) > 6:
        merged = merged[:6]

    while len(merged) < 3:
        longest_idx = max(range(len(merged)), key=lambda i: len(merged[i]))
        words = merged[longest_idx].split()
        mid = len(words) // 2
        merged[longest_idx:longest_idx+1] = [
            " ".join(words[:mid]),
            " ".join(words[mid:])
        ]

    return merged


# ── Prompt Engineering (heuristic only) ──────────────────────────────────────

def engineer_prompt(sentence: str, style: str) -> str:
    """Transform a sentence into a visually rich image generation prompt."""
    sentence = sentence.strip().rstrip(".")

    expansions = {
        "customer": "a satisfied customer in modern attire",
        "team": "a diverse professional team collaborating",
        "growth": "an upward trajectory symbolized by sprouting plants or rising graphs",
        "challenge": "a person at a crossroads facing a symbolic obstacle",
        "success": "a triumphant moment of achievement under golden light",
        "problem": "a puzzled expression, tangled wires or complex puzzle pieces",
        "solution": "a moment of clarity, a lightbulb illuminating a clean workspace",
        "data": "glowing holographic data streams and charts floating in air",
        "meeting": "professionals gathered around a sleek conference table",
        "launch": "a dramatic product reveal with spotlight and audience",
    }

    visual_sentence = sentence
    for word, expansion in expansions.items():
        visual_sentence = re.sub(rf'\b{word}\b', expansion, visual_sentence, flags=re.I, count=1)

    lighting = {
        "cinematic":   "dramatic side lighting, golden hour glow",
        "digital_art": "neon ambient lighting, vibrant atmosphere",
        "watercolor":  "soft natural daylight, pastel tones",
        "corporate":   "bright clean office lighting, neutral tones",
        "comic":       "high contrast bold shadows",
        "oil_painting":"warm candlelight, Rembrandt lighting",
    }

    style_desc = VISUAL_STYLES.get(style, VISUAL_STYLES["cinematic"])
    light = lighting.get(style, "dramatic lighting")

    return f"{visual_sentence}, {light}, {style_desc}, highly detailed, professional composition"


# ── Image Generation (Hugging Face only) ─────────────────────────────────────

def generate_image(prompt: str) -> tuple[bytes, str]:
    """Generate image using Hugging Face Router API with model fallback chain."""
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-2-1",
        "black-forest-labs/FLUX.1-schnell",
    ]

    for model in models:
        try:
            resp = requests.post(
                f"https://router.huggingface.co/hf-inference/models/{model}",
                headers={
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"inputs": prompt, "options": {"wait_for_model": True}},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.content, "jpeg"
            print(f"[!] {model} returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[!] {model} error: {e}")

    raise RuntimeError("All Hugging Face models failed. Check your token and permissions.")


def save_image(image_bytes: bytes, fmt: str, board_id: str, panel_idx: int) -> str:
    """Save image to disk and return the URL path."""
    filename = f"{board_id}_panel_{panel_idx}.{fmt}"
    (OUTPUT_DIR / filename).write_bytes(image_bytes)
    return f"/static/storyboards/{filename}"


# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Pitch Visualizer</title>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,400&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #f5f0e8;
      --ink: #1a1510;
      --muted: #8a7f70;
      --accent: #c8501a;
      --surface: #ede8de;
      --border: rgba(26,21,16,0.12);
      --radius: 12px;
    }

    body {
      background: var(--bg);
      color: var(--ink);
      font-family: 'DM Sans', sans-serif;
      min-height: 100vh;
      padding: 3rem 1.5rem 6rem;
    }

    .container { max-width: 900px; margin: 0 auto; }
    header { margin-bottom: 3.5rem; }

    .eyebrow {
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 0.5rem;
    }

    h1 {
      font-family: 'Fraunces', serif;
      font-size: clamp(2.4rem, 6vw, 4rem);
      font-weight: 600;
      line-height: 1.05;
      color: var(--ink);
    }

    h1 em { font-style: italic; font-weight: 300; color: var(--accent); }

    .subtitle {
      margin-top: 0.8rem;
      font-size: 1rem;
      color: var(--muted);
      line-height: 1.6;
      max-width: 540px;
    }

    .input-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      margin-bottom: 2rem;
    }

    label {
      display: block;
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.4rem;
    }

    textarea, select {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--ink);
      font-family: 'DM Sans', sans-serif;
      font-size: 0.95rem;
      padding: 0.8rem;
      outline: none;
      transition: border-color 0.2s;
      margin-bottom: 1rem;
    }

    textarea { min-height: 140px; resize: vertical; line-height: 1.6; }
    textarea:focus, select:focus { border-color: var(--accent); }

    select {
      cursor: pointer;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238a7f70' fill='none' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 0.8rem center;
      padding-right: 2.5rem;
    }

    .generate-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      width: 100%;
      padding: 1rem 2rem;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.15s;
    }

    .generate-btn:hover { opacity: 0.88; transform: translateY(-1px); }
    .generate-btn:disabled { opacity: 0.45; transform: none; cursor: not-allowed; }

    #storyboard { display: none; }

    .sb-header {
      display: flex;
      align-items: baseline;
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .sb-header h2 {
      font-family: 'Fraunces', serif;
      font-size: 1.6rem;
      font-weight: 300;
      font-style: italic;
    }

    .sb-meta {
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
    }

    .panels-grid { display: grid; gap: 2rem; }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      opacity: 0;
      transform: translateY(20px);
      transition: opacity 0.5s ease, transform 0.5s ease;
    }

    .panel.visible { opacity: 1; transform: translateY(0); }

    .panel-image-wrap {
      position: relative;
      width: 100%;
      padding-top: 56.25%;
      background: #ddd8cc;
      overflow: hidden;
    }

    .panel-image-wrap img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .panel-loading {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.75rem;
      color: var(--muted);
      font-size: 0.85rem;
    }

    .pulse-ring {
      width: 36px; height: 36px;
      border-radius: 50%;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .panel-body { padding: 1.25rem 1.5rem; }

    .panel-num {
      font-family: 'DM Mono', monospace;
      font-size: 0.68rem;
      color: var(--accent);
      letter-spacing: 0.15em;
      text-transform: uppercase;
      margin-bottom: 0.35rem;
    }

    .panel-caption { font-size: 1rem; line-height: 1.55; color: var(--ink); }

    .panel-prompt {
      margin-top: 0.75rem;
      padding: 0.6rem 0.8rem;
      background: var(--bg);
      border-left: 2px solid var(--border);
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
      line-height: 1.5;
      border-radius: 0 6px 6px 0;
    }

    .error-note {
      color: var(--accent);
      font-size: 0.85rem;
      margin-top: 0.5rem;
      font-family: 'DM Mono', monospace;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="eyebrow">AI Storyboard Generator</div>
    <h1>The Pitch <em>Visualizer</em></h1>
    <p class="subtitle">Paste a narrative. The engine segments it, engineers visual prompts, and generates a storyboard panel by panel.</p>
  </header>

  <div class="input-section">
    <label for="narrative">Your Narrative</label>
    <textarea id="narrative" placeholder="Paste a customer success story or sales narrative (3–6 sentences work best)…

Example: Sarah's small bakery was struggling to keep up with orders. She implemented our platform in a single afternoon. Within weeks, her order processing time dropped by 60%. Today, she serves twice as many customers with the same staff. Her story is one of hundreds we see every month."></textarea>

    <label for="style">Visual Style</label>
    <select id="style">
      <option value="cinematic">Cinematic Film Still</option>
      <option value="digital_art">Digital Concept Art</option>
      <option value="watercolor">Watercolor Illustration</option>
      <option value="corporate">Corporate Photography</option>
      <option value="comic">Graphic Novel / Comic</option>
      <option value="oil_painting">Classical Oil Painting</option>
    </select>

    <button class="generate-btn" id="genBtn" onclick="generate()">
      <span id="genLabel">Generate Storyboard</span>
      <div class="pulse-ring" id="genSpinner" style="display:none;width:20px;height:20px;border-width:2px;"></div>
    </button>
  </div>

  <div id="storyboard">
    <div class="sb-header">
      <h2>Your Storyboard</h2>
      <span class="sb-meta" id="sbMeta"></span>
    </div>
    <div class="panels-grid" id="panelsGrid"></div>
  </div>
</div>

<script>
async function generate() {
  const text = document.getElementById('narrative').value.trim();
  if (!text) { alert('Please enter a narrative.'); return; }

  const style = document.getElementById('style').value;
  const btn = document.getElementById('genBtn');
  const label = document.getElementById('genLabel');
  const spinner = document.getElementById('genSpinner');

  btn.disabled = true;
  label.textContent = 'Generating…';
  spinner.style.display = 'block';

  const grid = document.getElementById('panelsGrid');
  grid.innerHTML = '';
  document.getElementById('storyboard').style.display = 'block';

  try {
    const segRes = await fetch('/segment', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text })
    });
    const { segments, board_id } = await segRes.json();

    document.getElementById('sbMeta').textContent =
      `${segments.length} panels · ${style.replace('_',' ')} style`;

    segments.forEach((seg, i) => {
      grid.appendChild(createPanelSkeleton(i, seg));
    });

    for (let i = 0; i < segments.length; i++) {
      await generatePanel(board_id, segments[i], i, style);
    }

  } catch(e) {
    grid.innerHTML = `<p class="error-note">⚠ Error: ${e.message}</p>`;
  } finally {
    btn.disabled = false;
    label.textContent = 'Generate Storyboard';
    spinner.style.display = 'none';
  }
}

function createPanelSkeleton(idx, caption) {
  const div = document.createElement('div');
  div.className = 'panel';
  div.id = `panel-${idx}`;
  div.innerHTML = `
    <div class="panel-image-wrap">
      <div class="panel-loading" id="loading-${idx}">
        <div class="pulse-ring"></div>
        <span>Generating panel ${idx + 1}…</span>
      </div>
    </div>
    <div class="panel-body">
      <div class="panel-num">Panel ${idx + 1}</div>
      <div class="panel-caption">${caption}</div>
      <div class="panel-prompt" id="prompt-${idx}" style="display:none"></div>
    </div>`;
  setTimeout(() => div.classList.add('visible'), idx * 80);
  return div;
}

async function generatePanel(boardId, segment, idx, style) {
  const res = await fetch('/generate_panel', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ board_id: boardId, segment, panel_index: idx, style })
  });
  const data = await res.json();

  const loadingEl = document.getElementById(`loading-${idx}`);
  const wrap = loadingEl.parentElement;

  if (data.image_url) {
    const img = document.createElement('img');
    img.src = data.image_url;
    img.alt = `Panel ${idx + 1}`;
    img.style.opacity = '0';
    img.style.transition = 'opacity 0.4s ease';
    wrap.appendChild(img);
    img.onload = () => {
      loadingEl.style.display = 'none';
      img.style.opacity = '1';
    };
  } else {
    loadingEl.innerHTML = `<span style="color:#c8501a;font-size:0.8rem">⚠ ${data.error || 'Image generation failed'}</span>`;
  }

  const promptEl = document.getElementById(`prompt-${idx}`);
  if (data.engineered_prompt) {
    promptEl.textContent = '→ ' + data.engineered_prompt;
    promptEl.style.display = 'block';
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/segment", methods=["POST"])
def segment_route():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    segments = segment_text(text)
    board_id = uuid.uuid4().hex[:12]
    return jsonify({"segments": segments, "board_id": board_id})


@app.route("/generate_panel", methods=["POST"])
def generate_panel_route():
    data = request.get_json(force=True)
    segment = data.get("segment", "").strip()
    idx = int(data.get("panel_index", 0))
    style = data.get("style", "cinematic")
    board_id = data.get("board_id", uuid.uuid4().hex[:12])

    prompt = engineer_prompt(segment, style)

    try:
        image_bytes, fmt = generate_image(prompt)
        image_url = save_image(image_bytes, fmt, board_id, idx)
    except Exception as e:
        return jsonify({"error": str(e), "engineered_prompt": prompt}), 500

    return jsonify({
        "panel_index": idx,
        "segment": segment,
        "engineered_prompt": prompt,
        "image_url": image_url,
    })


# ── CLI Mode ──────────────────────────────────────────────────────────────────

def cli_mode():
    import sys
    print("\n=== Pitch Visualizer (CLI) ===")
    text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Enter narrative: ").strip()
    if not text:
        print("No text provided."); return

    style = input("Style [cinematic/digital_art/watercolor/corporate/comic/oil_painting] (default: cinematic): ").strip() or "cinematic"
    segments = segment_text(text)
    board_id = uuid.uuid4().hex[:12]
    print(f"\n  Segmented into {len(segments)} panels.\n")

    for i, seg in enumerate(segments):
        print(f"  Panel {i+1}: {seg[:80]}{'…' if len(seg)>80 else ''}")
        prompt = engineer_prompt(seg, style)
        print(f"    Prompt : {prompt[:100]}…")
        image_bytes, fmt = generate_image(prompt)
        url = save_image(image_bytes, fmt, board_id, i)
        print(f"    Saved  : {url}")

    print(f"\n  Done! Images saved to static/storyboards/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_mode()
    else:
        app.run(debug=True, port=5001)