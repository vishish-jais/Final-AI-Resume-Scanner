## AI Resume Scanner (Flask + SQLite) with Optional FastAPI Backend

This project provides a simple recruitment portal with an HR dashboard and a built‑in AI resume screening tool. It supports:

- HR and Candidate logins (Flask + SQLite)
- Resume uploads (PDF) and ATS-style screening
- Embedding-based similarity using Sentence Transformers
- Optional LLM (Llama-like) summaries when available, with robust fallbacks

---

## Quick Start (Windows PowerShell)

1) Create and activate a virtual environment, then install dependencies

```
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Run the Flask app

```
python app.py
```

Open http://127.0.0.1:5000

3) Default credentials (auto-seeded)

- HR: `hr` / `hr123`
- Candidate: `cand` / `cand123`

Uploads are stored in `uploads/`. App data is stored in `app.db` (SQLite).

---

## Resume Screening

- Navigate to `HR → Screening` or go to `http://127.0.0.1:5000/hr/screening`
- Paste a Job Description and upload a PDF resume
- First run may download the embedding model (`all-MiniLM-L6-v2`)

### How scoring works (see `ats_service.py`)

- Computes semantic similarity cosine using Sentence Transformers.
- Extracts a heuristic set of skills from both JD and Resume.
- Generates LLM summaries when the configured chat model is available; otherwise uses robust fallbacks to avoid empty fields.
- Business rules for the user-visible ATS Score:
  - If skill match fraction ≥ 90% → minimum score 95
  - If skill match fraction ≥ 75% → minimum score 90
  - If skill match fraction ≥ 50% OR cosine ≥ 0.5 → minimum score 80
  - If no skills match at all → score capped to 10 max

The UI shows:

- ATS Score and verdict
- Job Summary and Resume Summary
- Matched Skills and Missing Skills
- Semantic similarity (cosine)

> Note: If your PDF is a scanned image (no selectable text), extraction will be empty. Convert to text-based PDF or enable OCR.

---

## Optional: Use a Local LLM

By default, the app tries to use a Hugging Face chat model if available. Configure via environment variables before running the app:

```
$env:USE_OPENAI = "0"
$env:HF_CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # small, CPU-friendly
python app.py
```

Large, gated models like Llama 2/3 require `huggingface-cli login` and sufficient hardware.

---

## Backend API (Optional)

There is an optional FastAPI backend under `backend/` with an `/api/screen-resume` endpoint.

```
uvicorn backend.main:app --reload
```

Then POST form-data: `job_description`, `resume_file` to `http://127.0.0.1:8000/api/screen-resume`.

---

## Development

### Recommended .gitignore

```
venv/
__pycache__/
.env
*.db
uploads/
offload/
.DS_Store
```

### Troubleshooting

- Install errors: ensure you’re using the venv Python and `pip install -r requirements.txt`.
- `huggingface_hub.cached_download` import error: we pin compatible versions in `requirements.txt`.
- `PyPDF2` not found: ensure venv is active; we also install `pypdf` as a fallback.
- Empty summaries/skills: PDF may be image-only; switch to text-based PDF or add OCR fallback.

---

## License

MIT (see `LICENSE`).
