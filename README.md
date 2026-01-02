# Shutterstock Metadata Generator (Ollama)

Local Ollama vision models generate Shutterstock-ready metadata (description, keywords, categories, editorial flag, mature content, illustration) and append it to CSV.

## Prerequisites
- Python 3.10+ and Ollama running with a vision model (tested on `llama3.2-vision`).
- Install deps (prefer a venv): `python -m pip install -r requirements.txt`

## Quick Start
Directory run (recommended defaults):
```bash
python3 image_analyzer.py \
  --dir path/to/images \
  --csv shutterstock.csv \
  --base-url http://localhost:11434/ \
  --model llama3.2-vision \
  --num-predict 800 --top-k 150 --max-retries 3 \
  --resize-max 1024 --resize-quality 85
```
Single image:
```bash
python3 image_analyzer.py --image path/to/img.jpg --csv out.csv --base-url http://localhost:11434/
```
Options:
- `--recursive` to include subfolders.
- `--prompt-file` to override the default prompt.
- `--hint` to add context for all images.
- `--no-fallback` disables the safer second pass on failures; `--max-retries` sets initial attempts (fallback uses cooler options).
- `--no-progress` for quiet output.

## Prompt (summary)
- Caption: one concise sentence <200 chars, no filler/ellipsis/repeated adjectives; prefer exact names when clear.
- Keywords: 7–50 unique, no placeholders/HTML/“...”, max 30 after dedup.
- Categories: 1–2 from the fixed list only.
- Booleans: `editorial`, `mature_content`, `illustration`.

## Features
- Keyword post-processing (dedup, drop junk, limit length) and category validation.
- Optional resizing/caching (`.cache/resized`) to improve stability/speed on large images.
- Fallback pass with safer generation options for initial failures.
- Prompt compliance checker for generated CSVs.

## Testing
```bash
python3 -m unittest tests/test_image_analyzer.py
```

## Notes
- Generated artifacts (`shutterstock.csv`, `.cache/resized/`, sample image folders) are git-ignored by default.
- Use `--no-fallback` or different options if you need to mirror upstream behavior exactly.
