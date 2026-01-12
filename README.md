# Reliable Multimodal Math Mentor

An AI Engineer Assignment submission. This application solves math problems using a Multi-Agent system (Parser, Router, Solver, Verifier, Explainer) backed by RAG and Memory.

## Features
- **Multimodal:** Accepts Text, Images, and Audio files.
- **RAG:** Retrieves formulas from a trusted Knowledge Base.
- **HITL:** Includes a "Verify Extraction" step for human approval.
- **Memory:** Caches solutions to learn over time.

## Setup
1. `pip install -r requirements.txt`
2. Add `GEMINI_API_KEY` to `.env`
3. `streamlit run app.py`