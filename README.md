# Code-Debugging-RAG-Agent

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** workflow that uses [Marqo](https://github.com/marqo-ai/marqo) as the vector store for code documents and [Ollama](https://github.com/jmorganca/ollama) as the Large Language Model (LLM) generator. The pipeline leverages [Langchain Community](https://github.com/langchain4j/langchain-community) wrappers to seamlessly integrate both Marqo and Ollama.

This repository demonstrates a **Retrieval‑Augmented Generation (RAG)** workflow that can run **two interchangeable retrieval stacks**:

| Retrieval backend | File | Vector index | Default algorithm |
|-------------------|------|--------------|-------------------|
| **Marqo** (service) | `rag_debugger.py` | `code_files` index inside Marqo | IVF + PQ |
| **FAISS** (in‑process) | `rag_faiss_debugger.py` | `faiss_hnsw.pkl` | HNSW |

For generation we leverage [Ollama](https://github.com/jmorganca/ollama) **locally‑served LLMs** (e.g. `llama3.2:1b`).  Both integrations are wired up through [LangChain Community](https://python.langchain.com/).

A thin [FastAPI](https://fastapi.tiangolo.com/) micro‑service (`api.py`) exposes a single `/debug` endpoint; once it produces the fix it **emails the result** to you.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup & Usage](#setup--usage)
  - [1 / Run Marqo (optional)](#1-run-marqo-optional)
  - [2 / Index the Codebase](#2-index-the-codebase)
  - [3 / Configure Ollama](#3-configure-ollama)
  - [4 / Populate `.env`](#4-populate-env)
  - [5 / Start the API](#5-start-the-api)
  - [6 / Send a Log & Get Email](#6-send-a-log--get-email)
- [Key Components](#key-components)
- [How It Works](#how-it-works)
- [Benchmarking & Energy Metrics](#benchmarking--energy-metrics)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview
Feed the system an **exception stack‑trace** (plain text).  
The agent will:

1. **Retrieve** the most relevant code snippets using Marqo **or** FAISS.  
2. **Compose** a structured prompt (error info + code context).  
3. **Generate** a debugging recommendation with an Ollama LLM.  
4. **Respond** via HTTP JSON *and* **email you** the same solution.

---

## Features
| Category | Highlights |
|----------|------------|
| **Retrieval** | Choice of Marqo (service) *or* FAISS (in‑process). |
| **LLM** | Any Ollama model (`llama2:7b`, `llama3.2:1b`, `phi3-mini`, …). |
| **FastAPI** | `POST /debug` accepts a `.txt` log and returns JSON. |
| **Email alerts** | SMTP credentials in `.env`; solution mailed automatically. |
| **Energy benchmarking** | `ollama_benchmark_clean.py` measures time, Δ‑RAM, CO₂, tokens/s. |
| **Pluggable indexes** | IVF + PQ (Marqo default) vs. HNSW (FAISS). |


## Project Structure
```text
RAG_PROJECT/
├── api.py                     # FastAPI service (uses FAISS by default)
├── rag_debugger.py            # Marqo RAG pipeline
├── rag_faiss_debugger.py      # FAISS RAG pipeline
├── mailer.py                  # send_email()
├── marqodb_client.py          # helper to (re)index Marqo
├── benchmark_models.py        # quick LLM benchmark
├── ollama_benchmark_clean.py  # energy‑efficiency benchmark
├── codebase/                  # code snippets (.txt) to be indexed
├── faiss_hnsw.pkl             # FAISS index (auto‑generated)
├── emissions.csv              # CodeCarbon logs
├── agno_demo_app_error.log    # sample error log
```
---

## 🛠️ Installation

```bash
git clone https://github.com/your‑username/code‑debugging‑rag‑agent.git
cd code‑debugging‑rag‑agent
```

## ⚙️ Setup & Usage
1) Run Marqo (optional)
If you plan to use the Marqo retrieval path:
```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
```

2) Configure Ollama
```bash
ollama pull llama3.2:1b    # or any model you prefer
ollama serve
```
3) Populate .env
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password      # Gmail App Password
EMAIL_FROM="AI Debugger <your_email@gmail.com>"
RECIPIENT=recipient@email.com    # address to receive fixes
```

4) Start the API
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```
5) Send a Log & Get Email
```bash
curl -F "file=@agno_demo_app_error.log;type=text/plain" \
     http://localhost:8000/debug
```
