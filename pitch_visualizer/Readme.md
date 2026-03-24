# 🖼 The Pitch Visualizer

> Paste a narrative. Get a visual storyboard. Powered by AI prompt engineering + Hugging Face image generation.

---

## What It Does

The Pitch Visualizer automates the creative bottleneck of turning a written narrative into a storyboard:

1. **Segments** your narrative into 3–6 logical scene panels (using NLTK tokenization + smart merging of short sentences)
2. **Engineers visual prompts** for each panel — transforms abstract prose into rich, visually descriptive image prompts
3. **Generates an image** per panel via Hugging Face (SDXL → SD v2 → FLUX fallback chain)
4. **Presents** a polished, interactive storyboard in the browser, with panels appearing one-by-one as they generate

---

## Setup

### Prerequisites
- Python 3.10+
- A Hugging Face account with an API token

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/pitch-visualizer.git
cd pitch-visualizer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### API Key Configuration

**Step 1** — Create a `.env` file in the project root:

```
HF_API_KEY=hf_your_token_here
```

**Step 2** — Get your Hugging Face token (with the right permissions):

1. Go to **huggingface.co → Settings → Access Tokens**
2. Click **New Token**
3. Enable ✅ **Make calls to Inference Providers**
4. Enable ✅ **Read**
5. Copy the token and paste it into `.env`

> ⚠️ The "Make calls to Inference Providers" permission is required. A token with only "Read" access will return a 403 error.

---

## Usage

### Web Interface (recommended)

```bash
python app.py
```

Open **http://localhost:5001**. Paste your narrative (3–6 sentences), choose a visual style, and click **Generate Storyboard**. Panels appear progressively as each image finishes generating.

### CLI Mode

```bash
# Interactive (prompts for input)
python app.py cli

# Pass text inline
python app.py cli "Sarah's bakery was struggling. She adopted our platform. Orders doubled in a month."
```

CLI mode saves images to `static/storyboards/` and generates a standalone `storyboard.html` file.

---

## Visual Styles

| Style | Description |
|---|---|
| Cinematic Film Still | Dramatic lighting, shallow DOF, photorealistic |
| Digital Concept Art | Vibrant, detailed illustration, ArtStation aesthetic |
| Watercolor Illustration | Soft washes, organic textures, painterly |
| Corporate Photography | Clean, bright, professional lifestyle |
| Graphic Novel / Comic | Bold ink outlines, cel-shaded, dynamic |
| Classical Oil Painting | Rembrandt lighting, impasto, Old Masters |

Visual consistency is maintained by appending the selected style's keyword string to every panel prompt — ensuring all images share the same artistic treatment.

---

## Design Choices

### Narrative Segmentation
We use NLTK `sent_tokenize` (Punkt tokenizer) with a post-processing step:
- Short sentences (< 6 words) are merged with their neighbour to avoid trivial panels
- Very long texts are capped at 6 panels for UX clarity
- If fewer than 3 segments result, the longest segment is split at its midpoint

### Prompt Engineering
Simply using the original sentence as the image prompt produces poor results — it's too abstract and text-centric. Our heuristic approach:
- A keyword-expansion dictionary maps common business/story words to visual descriptions (e.g., `"customer"` → `"a satisfied customer in modern attire"`)
- Lighting descriptors and style keywords are appended based on the selected visual style
- This transforms prose into spatially-described scenes with mood and composition cues

### Image Generation
Uses Hugging Face's router API (`router.huggingface.co`) with a three-model fallback chain:
1. `stabilityai/stable-diffusion-xl-base-1.0` (best quality)
2. `stabilityai/stable-diffusion-2-1` (faster)
3. `black-forest-labs/FLUX.1-schnell` (most reliable)

### Progressive Panel Rendering
Panels are created as skeleton elements immediately, then filled asynchronously as each image returns. This avoids a blank wait and gives the feel of a "live" generation.

---

## Project Structure

```
pitch-visualizer/
├── app.py                  # Main application (Flask + CLI)
├── requirements.txt
├── .env                    # Your HF_API_KEY goes here (never commit this!)
├── .gitignore
├── static/
│   └── storyboards/        # Generated images (created at runtime)
└── README.md
```

---

## Bonus Features Implemented

- ✅ **Visual Consistency**: Every prompt appends the selected style string
- ✅ **User-Selectable Styles**: 6 distinct visual styles selectable from the UI
- ✅ **Heuristic Prompt Engineering**: Keyword expansion transforms prose into visual briefs
- ✅ **Dynamic UI**: Panels appear progressively as images generate
- ✅ **Multi-model Fallback**: SDXL → SD v2 → FLUX
- ✅ **CLI + Web**: Both interfaces from the same codebase