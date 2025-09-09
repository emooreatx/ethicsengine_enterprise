# Brad's Deplomynet Steps

End-to-end steps to pull, set up, and run EthicsEngine locally. All commands are copy-paste ready for macOS/Linux terminals using zsh/bash.

## Prerequisites

- Git
- Python 3.10+ (recommended)
- pip
- Optional (for local LLM via Ollama): Ollama running at http://127.0.0.1:11434
- Optional (for OpenAI): An OpenAI API key

Known ports:
- API (FastAPI): 8000 (local dev), 8080 (Docker)
- UI (Streamlit): 8501

---

## 1) Pull the repository

```bash
git clone https://github.com/emooreatx/ethicsengine_enterprise EthicsEnterprise
cd EthicsEnterprise
```

---

## 2) Local development setup (venv)

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3) Start the API (FastAPI + Uvicorn)

In terminal 1:

```bash
cd EthicsEnterprise
source .venv/bin/activate
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify:

- API health: http://localhost:8000/health  
- API docs: http://localhost:8000/docs

---

## 4) Start the UI (Streamlit)

In terminal 2:

```bash
cd EthicsEnterprise
source .venv/bin/activate
.venv/bin/streamlit run ui/app.py
```

Open the UI: http://localhost:8501

Notes:
- The UI default API Base URL is set to http://127.0.0.1:8000
- You can change the API Base URL at runtime via the UI sidebar (Server Status → Server Configuration)

---

## 5) Run a pipeline

Option A: From the UI
- Go to the “Dashboard” page in the UI
- Select a pipeline from the list
- Click “Run Pipeline”
- Monitor progress on “Run Monitoring”

Option B: From the CLI
```bash
cd EthicsEnterprise
source .venv/bin/activate
.venv/bin/python run_pipeline.py bench_q1
```

Results are written to:
```bash
ls results
```
View one:
```bash
cat results/<run_id>.json
```

---

## 6) LLM configuration options

The project supports two primary backends: Ollama or OpenAI. Default is Ollama (local).

### A) Use Ollama (default)
- Default base URL: http://127.0.0.1:11434/v1
- Default model: gemma3:4b-it-q8_0

Ensure Ollama is running. In the UI sidebar:
- Choose “Ollama”
- Verify models are listed
- Select or manually enter a model (e.g., `gemma3:4b-it-q8_0`)
- Click “Set Ollama Configuration”

### B) Use OpenAI
Set your API key and choose a model in the UI, or export the key via terminal.

Terminal:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

UI (sidebar):
- Choose “OpenAI”
- Pick a model (e.g., gpt-4o-mini)
- Click “Set OpenAI Configuration”

Advanced (optional): Provide a full config list via env variable consumable by autogen:
```bash
export ETHICSENGINE_CONFIG_LIST='[{"api_type":"openai","model":"gpt-4o-mini","api_key":"'"$OPENAI_API_KEY"'","base_url":"https://api.openai.com/v1"}]'
```

---

## 7) Optional: Run the API with Docker (backend only)

This runs the FastAPI backend in a container exposing port 8080. The Streamlit UI can still run locally and point to 8080.

Build and run:
```bash
cd EthicsEnterprise
docker compose up --build -d
```

Check health:
```bash
curl http://localhost:8080/health
```

In the Streamlit UI (http://localhost:8501), set API Base URL to:
```
http://127.0.0.1:8080
```

Stop containers:
```bash
docker compose down
```

---

## Troubleshooting

- UI cannot connect to API:
  - Ensure the API is running and healthy:
    ```bash
    curl http://localhost:8000/health
    ```
  - In UI sidebar, confirm API Base URL matches the API port (8000 for local, 8080 for Docker).

- Ollama not listing models:
  - Ensure Ollama is running locally and reachable at http://127.0.0.1:11434
  - Manually enter a known model (e.g., `gemma3:4b-it-q8_0`) in the UI and click “Set Ollama Configuration”.

- OpenAI requests failing:
  - Ensure `OPENAI_API_KEY` is exported:
    ```bash
    export OPENAI_API_KEY="sk-your-key-here"
    ```
  - Switch the UI to “OpenAI” and set the model, then click “Set OpenAI Configuration”.

- Permissions / venv issues:
  - Recreate the venv:
    ```bash
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

---

## Stop & Cleanup

Stop Streamlit and Uvicorn:
- Press Ctrl+C in each terminal

Deactivate venv:
```bash
deactivate
```

Stop Docker backend (if used):
```bash
docker compose down
```

---

## One-time quick run summary

```bash
# Clone and enter
git clone https://github.com/emooreatx/ethicsengine_enterprise EthicsEnterprise
cd EthicsEnterprise

# Venv + deps
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# API (terminal 1)
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# UI (terminal 2)
cd EthicsEnterprise
source .venv/bin/activate
.venv/bin/streamlit run ui/app.py
```

- API: http://localhost:8000/health
- UI: http://localhost:8501
- Run a pipeline from the UI or:
```bash
.venv/bin/python run_pipeline.py bench_q1
```
